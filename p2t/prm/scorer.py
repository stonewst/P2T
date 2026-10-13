"""Process-reward scorer: PRM forward -> per-step reward -> policy tokens.

Reads the PRM's score at each separator position and broadcasts it onto the policy tokens
of that step. With `use_p2t_attribution` on it additionally runs a backward pass to get a
per-token attribution score I_i (see prm/attribution.py) and maps it onto the policy
tokens. The P2T token reward itself is built on the driver (`_compute_p2t_reward` in
trainer/ppo/ray_trainer.py), after repeat calibration may have modified R_s.

A step is one `\n\n` segment, or, with `use_pack_short_process`, a block of consecutive
short segments (see prm/packing.py). The PRM input is a chat-templated conversation whose
assistant content is `sep.join(steps) + sep`; R_s is read from the 2-class head at each
`sep` (see `reward_transform`).

Per response, `score` returns:
    process_reward   (n_resp,) float32   step reward R_s broadcast to the step's tokens
    process_step_idx (n_resp,) int64     owning step of each token (-1 = none)
    process_info     list[dict]          {process_text, process_reward, process_logits}
    p2t_influence    (n_resp,) float32   signed attribution score I_i; zero when
                                         attribution is off

Pipeline (stages in `score`):
    strip special tokens -> char spans and text -> split into `\n\n` segments
    -> pack into blocks -> build PRM input -> PRM forward, read R_s at each sep
    -> broadcast R_s to policy tokens (+ map I_i) -> EOS inherits the last block's R_s.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from p2t.prm.align import (
    conversation_step_spans,
    owning_segment,
    response_step_spans,
    token_char_spans,
    transfer_influence,
)
from p2t.prm.attribution import ATTRIBUTION_TARGETS, BACKWARD_MODES, attribution_mode, prm_token_influence
from p2t.prm.packing import PACK_ALGORITHMS, pack_blocks

logger = logging.getLogger(__name__)


class ProcessRewardScorer:
    """Loads a PRM once and scores (problem, response) pairs into per-token process rewards."""

    def __init__(
        self,
        prm_path: str,
        policy_tokenizer,
        sep: str = "<extra_0>",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        system_prompt: Optional[str] = None,   # PRM system message; None = template default
        max_prm_len: int = 0,                  # 0 = no cap; otherwise skip longer sequences
        attn_implementation: str = "sdpa",     # sdpa | flash_attention_2 | eager
        positive_class: int = 1,               # class index treated as "correct"
        reward_transform: str = "prob_diff",   # "prob_diff": p_pos - p_neg in [-1, 1]
                                               # "prob":      p_pos in [0, 1]
        use_pack_short_process: bool = False,       # enable packing
        pack_min_token_per_process: int = 32,       # packing floor (a shorter block is penalised)
        pack_max_token_per_process: int = 128,      # packing ceiling
        pack_algorithm: str = "dp",                 # dp | greedy
        pack_over_penalty: float = 1.0,             # weight of the ceiling penalty vs the floor
        # token attribution (P2T); adds one backward per response
        use_p2t_attribution: bool = False,
        p2t_attribution_target: str = "logit_diff",     # logit_diff | logprob | prob
        p2t_attribution_backward: str = "single",  # single | per_step
        p2t_attribution_grad_ckpt: bool = True,  # trade compute for activation memory
    ):
        self.sep = sep
        self.device = device
        self.policy_tok = policy_tokenizer
        self.max_prm_len = int(max_prm_len)
        self.system_prompt = system_prompt or None   # "" -> None
        self.positive_class = int(positive_class)
        assert reward_transform in ("prob_diff", "prob"), reward_transform
        self.reward_transform = reward_transform

        # packing
        self.use_pack_short_process = bool(use_pack_short_process)
        self.pack_min_token_per_process = int(pack_min_token_per_process)
        self.pack_max_token_per_process = int(pack_max_token_per_process)
        self.pack_algorithm = pack_algorithm
        self.pack_over_penalty = float(pack_over_penalty)
        if self.use_pack_short_process:
            if pack_algorithm not in PACK_ALGORITHMS:
                raise ValueError(f"pack_algorithm={pack_algorithm!r}, want one of {PACK_ALGORITHMS}")
            if self.pack_min_token_per_process <= 0:
                raise ValueError("use_pack_short_process=True needs pack_min_token_per_process > 0")
            if self.pack_max_token_per_process < self.pack_min_token_per_process:
                raise ValueError(f"pack_max_token_per_process ({self.pack_max_token_per_process}) < "
                                 f"pack_min_token_per_process ({self.pack_min_token_per_process})")

        # token attribution (P2T)
        self.use_p2t_attribution = bool(use_p2t_attribution)
        if p2t_attribution_target not in ATTRIBUTION_TARGETS:
            raise ValueError(f"p2t_attribution_target={p2t_attribution_target!r}, want one of {ATTRIBUTION_TARGETS}")
        if p2t_attribution_backward not in BACKWARD_MODES:
            raise ValueError(f"p2t_attribution_backward={p2t_attribution_backward!r}, want one of {BACKWARD_MODES}")
        self.p2t_attribution_target = p2t_attribution_target
        self.p2t_attribution_backward = p2t_attribution_backward
        self.p2t_attribution_grad_ckpt = bool(p2t_attribution_grad_ckpt)

        # The PRM's own tokenizer (usually different from the policy's).
        self.prm_tok = AutoTokenizer.from_pretrained(prm_path, trust_remote_code=True)
        # sdpa / flash attention avoid the O(L^2) memory of eager attention.
        self.prm = AutoModel.from_pretrained(
            prm_path, device_map=device, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).eval()
        # Freeze all parameters; attribution only needs gradients w.r.t. inputs_embeds.
        for p in self.prm.parameters():
            p.requires_grad_(False)

        # sep (e.g. "<extra_0>") must encode to a single token.
        self.sep_id = self.prm_tok.encode(sep)[0]
        self.pad_id = self.prm_tok.pad_token_id
        # Attribution uses the pad embedding as its baseline.
        if self.use_p2t_attribution and self.pad_id is None:
            raise ValueError(
                f"use_p2t_attribution=True but the PRM tokenizer at {prm_path!r} has no "
                "pad_token_id; I_i is measured against the pad embedding as its baseline"
            )

    # ---------------------------------------------------------------- placement
    def to_device(self, device: str):
        """Move the PRM between CPU and GPU."""
        if device == self.device:
            return self
        self.prm.to(device)
        self.device = device
        if device == "cpu":
            torch.cuda.empty_cache()
        return self

    # ---------------------------------------------------------------- helpers
    def _split_text_ids(self, response_ids: Sequence[int]):
        """Strip special tokens (e.g. EOS) before char alignment.

        Returns:
            keep_idx: positions of the retained tokens in `response_ids`.
            keep_ids: the retained token ids.
        """
        special = set(self.policy_tok.all_special_ids or [])
        keep_idx, keep_ids = [], []
        for i, t in enumerate(response_ids):
            if int(t) in special:
                continue
            keep_idx.append(i)
            keep_ids.append(int(t))
        return keep_idx, keep_ids

    def _pack(self, segments, seg_of_token):
        """Merge `\\n\\n` segments into blocks (one PRM step each).

        Segment lengths are counted with `seg_of_token`, the same ownership used for N_s.

        Returns: [(start, end), ...] half-open segment-index ranges covering 0..len(segments).
        """
        if not self.use_pack_short_process or len(segments) < 2:
            return [(k, k + 1) for k in range(len(segments))]
        # Tokens per segment, including its trailing `\n\n`.
        placed = seg_of_token[seg_of_token >= 0]
        n_tok = np.bincount(placed, minlength=len(segments))
        return pack_blocks(
            n_tok,
            min_tok=self.pack_min_token_per_process,
            max_tok=self.pack_max_token_per_process,
            over=self.pack_over_penalty,
            algorithm=self.pack_algorithm,
        )

    def _read_reward(self, probs: torch.Tensor, position: int) -> float:
        """Read a scalar score from the PRM's binary classification head at `position`."""
        pos = self.positive_class
        if self.reward_transform == "prob_diff":
            return float(probs[position, pos] - probs[position, 1 - pos])   # [-1, 1]
        if self.reward_transform == "prob":
            return float(probs[position, pos])                              # [0, 1]
        raise ValueError(f"unknown reward_transform: {self.reward_transform!r}")

    # ---------------------------------------------------------------- scoring
    # Not decorated with @torch.no_grad(): attribution needs gradients.
    def score(self, problem: str, response_ids: Sequence[int]) -> Dict:
        """Score a single response.

        Args:
            problem:      raw problem text; the PRM's chat template is applied here.
            response_ids: policy token ids of the response, without padding.
        Returns: see the module docstring.
        """
        # ---- Stage 0: result for responses that cannot be scored ----
        n = len(response_ids)

        def _empty() -> Dict:
            """Return the "not scored" result."""
            return {
                "process_reward": np.zeros(n, dtype=np.float32),
                "process_step_idx": np.full(n, -1, dtype=np.int64),
                "process_info": [],
                "coverage": 0.0,
                "skipped": True,
                "n_steps": 0,
                "n_segments": 0,
                "n_blocks": 0,
                "seg_tokens": np.zeros(0, dtype=np.int64),   # per-segment token count (pre-pack)
                "blk_tokens": np.zeros(0, dtype=np.int64),   # per-block token count (post-pack)
                "p2t_influence": np.zeros(n, dtype=np.float32),
            }
        if n == 0:
            return _empty()

        # ---- Stage 1: text, char spans and `\n\n` segments ----
        keep_idx, keep_ids = self._split_text_ids(response_ids)
        if not keep_ids:
            return _empty()
        policy_spans, response = token_char_spans(self.policy_tok, keep_ids)
        if not response.strip():
            return _empty()

        # Empty segments are kept so that "\n\n".join(segments) == response.
        segments = response.split("\n\n")
        seg_spans = response_step_spans(segments)
        seg_of_token = owning_segment(policy_spans, seg_spans)

        # Drop trailing empty segments so the last block has content (EOS inherits its R_s).
        # Internal empty segments are kept to preserve the join identity.
        n_keep = len(segments)
        while n_keep > 1 and segments[n_keep - 1] == "":
            n_keep -= 1
        if n_keep < len(segments):
            segments = segments[:n_keep]
            # Separator tokens of the dropped segments go to the last retained segment.
            seg_of_token = np.where(seg_of_token >= n_keep, n_keep - 1, seg_of_token)

        # ---- Stage 2: pack segments into blocks ----
        blocks = self._pack(segments, seg_of_token)               # [(seg_start, seg_end), ...]
        steps = ["\n\n".join(segments[a:b]) for a, b in blocks]
        # seg_to_block: e.g. blocks=[(0,2),(2,3)] -> [0,0,1].
        seg_to_block = np.empty(len(segments), dtype=np.int64)
        for bi, (a, b) in enumerate(blocks):
            seg_to_block[a:b] = bi
        # token -> block; -1 stays -1.
        block_of_token = np.where(seg_of_token >= 0,
                                  seg_to_block[np.clip(seg_of_token, 0, None)], -1)

        # ---- Stage 3: build the PRM input (one sep after each block) ----
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages += [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": self.sep.join(steps) + self.sep},
        ]
        conversation = self.prm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prm_ids = self.prm_tok.encode(conversation, add_special_tokens=False)
        if self.max_prm_len and len(prm_ids) > self.max_prm_len:
            return _empty()

        # ---- Stage 3.5: PRM-side char spans (attribution only) ----
        # Spans are computed on the per-token decoded text so they stay self-consistent.
        prm_spans = conv_spans = None
        response_start = 0
        if self.use_p2t_attribution:
            prm_spans, prm_text = token_char_spans(self.prm_tok, prm_ids)
            conv_spans = conversation_step_spans(prm_text, steps, self.sep)
            if not conv_spans:
                return _empty()
            # First PRM token of block 0; earlier (prompt) tokens get no influence.
            response_start = next(
                (t for t, (_, c1) in enumerate(prm_spans) if c1 > conv_spans[0][0]), 0
            )

        # ---- Stage 4: PRM forward (+ backward), read R_s at each sep ----
        input_ids = torch.tensor([prm_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        # The assistant turn is last, so keep the last len(steps) separators in case sep also
        # appears in the problem.
        sep_positions = (input_ids[0] == self.sep_id).nonzero(as_tuple=True)[0].tolist()
        if len(sep_positions) > len(steps):
            sep_positions = sep_positions[-len(steps):]
        if len(sep_positions) != len(steps) or not sep_positions:
            return _empty()

        if self.use_p2t_attribution:
            # R_s and I_i come from the same forward pass.
            with attribution_mode(self.prm, self.p2t_attribution_grad_ckpt):
                prm_influence, logits = prm_token_influence(
                    self.prm, input_ids, attention_mask, sep_positions,
                    pad_id=self.pad_id,
                    target=self.p2t_attribution_target,
                    backward_mode=self.p2t_attribution_backward,
                    positive_class=self.positive_class,
                    response_start=response_start,
                )
        else:
            prm_influence = None
            with torch.no_grad():
                outputs = self.prm(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()[0]         # (L, num_classes)

        probs = torch.softmax(logits, dim=-1)
        block_rewards = [self._read_reward(probs, sp) for sp in sep_positions]
        block_logits = [logits[sp].cpu().tolist() for sp in sep_positions]

        # ---- Stage 5: broadcast R_s to the tokens of each block ----
        proc = np.zeros(n, dtype=np.float32)               # per-token process reward
        step_idx = np.full(n, -1, dtype=np.int64)          # per-token block index (-1 = none)
        n_covered = 0
        for j, i in enumerate(keep_idx):                   # j: content index, i: response index
            k = int(block_of_token[j])
            if 0 <= k < len(block_rewards):
                proc[i] = block_rewards[k]
                step_idx[i] = k
                n_covered += 1

        # ---- Stage 5b: map I_i from PRM tokens to policy tokens ----
        p2t_influence = np.zeros(n, dtype=np.float32)
        if prm_influence is not None:
            infl_keep = transfer_influence(
                prm_spans=prm_spans,
                prm_influence=prm_influence.cpu().numpy(),
                conv_spans=conv_spans,
                resp_spans=response_step_spans(steps),
                policy_spans=policy_spans,
                policy_step_idx=block_of_token,
            )
            p2t_influence[np.asarray(keep_idx, dtype=np.int64)] = infl_keep.astype(np.float32)

        # ---- Stage 6: special tokens (e.g. EOS) inherit the last block's R_s ----
        # Giving them 0 would penalize stopping. They keep step_idx = -1.
        special = np.ones(n, dtype=bool)
        special[keep_idx] = False
        if special.any() and block_rewards:
            proc[special] = block_rewards[-1]

        process_info = [
            {"process_text": text, "process_reward": reward, "process_logits": lg}
            for text, reward, lg in zip(steps, block_rewards, block_logits)
        ]

        # Per-segment / per-block token counts (separators count toward their owner).
        seg_tokens = np.bincount(seg_of_token[seg_of_token >= 0],
                                 minlength=len(segments)).astype(np.int64)
        blk_tokens = np.bincount(block_of_token[block_of_token >= 0],
                                 minlength=len(blocks)).astype(np.int64)

        # ---- Stage 7: return ----
        return {
            "process_reward": proc,          # (n,) per-token R_s
            "process_step_idx": step_idx,    # (n,) per-token block index
            "process_info": process_info,    # per-block {text, R_s, logits}
            # Fraction of content tokens matched to a block; < 1 indicates an alignment issue.
            "coverage": n_covered / max(len(keep_idx), 1),
            "skipped": False,
            "n_steps": len(steps),           # #PRM steps (== #blocks)
            "n_segments": len(segments),     # #finest "\n\n" segments
            "n_blocks": len(blocks),         # #blocks after packing (== n_segments when packing is off)
            "seg_tokens": seg_tokens,        # (n_segments,) per-segment token count (pre-pack)
            "blk_tokens": blk_tokens,        # (n_blocks,)   per-block token count (post-pack)
            "p2t_influence": p2t_influence,  # (n,) signed I_i; zero when attribution is off
        }

    def score_batch(self, problems: Sequence[str],
                    responses_ids: Sequence[Sequence[int]]) -> List[Dict]:
        """Score responses one at a time, clearing the CUDA cache after each."""
        out = []
        for prob, ids in zip(problems, responses_ids):
            out.append(self.score(prob, ids))
            torch.cuda.empty_cache()
        return out
