"""
In-process P2T scorer: PRM forward/backward -> token influence -> policy-token rewards.

Ties together:
  p2t_attribution.compute_p2t_token_rewards   (PRM-side influence + step rewards)
  p2t_align.align_prm_influence_to_policy     (PRM tokens -> policy tokens)

Produces, per response, the batch fields agreed for the P2T advantage:
  p2t_token_reward      (n_resp,)  Eq.(3) reward on policy tokens
  process_token_reward  (n_resp,)  step reward R_s broadcast to policy tokens
  p2t_influence         (n_resp,)  token influence I_i on policy tokens
  p2t_step_idx          (n_resp,)  which process each policy token belongs to (-1 = none)
  process_info          list[dict] {process_text, process_reward, process_logits}

The class holds no distributed state, so it can be driven either directly (tests)
or from inside a data-parallel Ray worker. When colocated with the training
workers the PRM rests on CPU and is moved onto the GPU only for scoring; see
`to_device`.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2t_align import align_prm_influence_to_policy, token_char_spans
from p2t_attribution import P2TConfig, compute_p2t_token_rewards


class P2TScorer:
    """Loads a PRM once and scores (problem, response) pairs into token-level rewards."""

    def __init__(
        self,
        prm_path: str,
        policy_tokenizer,
        sep: str = "<extra_0>",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attribution_target: str = "logit_diff",
        backward_mode: str = "single",
        temperature: float = 1.0,
        process_reward_scale: str = "null",   # null ([0,1]) | signed (2R-1 -> [-1,1])
        max_prm_len: int = 0,          # 0 = no cap; otherwise skip longer sequences
        grad_checkpointing: bool = False,
        attn_implementation: str = "sdpa",   # sdpa | flash_attention_2 | eager
    ):
        self.sep = sep
        self.device = device
        self.policy_tok = policy_tokenizer
        self.max_prm_len = max_prm_len
        process_reward_scale = process_reward_scale or "null"   # yaml `null` -> None
        assert process_reward_scale in ("null", "signed"), process_reward_scale
        self.process_reward_scale = process_reward_scale

        self.prm_tok = AutoTokenizer.from_pretrained(prm_path, trust_remote_code=True)
        # eager attention materialises the full L x L matrix (O(L^2) memory): ~73GB
        # at 3k tokens, OOM at 8k. sdpa/flash keep it O(L) (~29GB at 3k) with
        # bf16-identical logits, so sdpa is the default.
        self.prm = AutoModel.from_pretrained(
            prm_path, device_map=device, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).eval()
        # attribution differentiates w.r.t. the input embeddings only, never the weights
        for p in self.prm.parameters():
            p.requires_grad_(False)

        if grad_checkpointing:
            # trades a full recompute per backward for a large drop in activation
            # memory; only worth it for very long sequences. use_reentrant=False is
            # required because no parameter carries requires_grad.
            self.prm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.sep_id = self.prm_tok.encode(sep)[0]
        self.pad_id = self.prm_tok.pad_token_id
        self.cfg = P2TConfig(
            attribution_target=attribution_target,
            backward_mode=backward_mode,
            temperature=temperature,
        )
        self.cfg.validate()

    # ---------------------------------------------------------------- placement
    def to_device(self, device: str):
        """Move the PRM between CPU and GPU.

        When the scorer is colocated with the training workers the PRM must not
        hold GPU memory outside of the scoring window, otherwise it competes with
        the rollout engine's KV cache.
        """
        if device == self.device:
            return self
        self.prm.to(device)
        self.device = device
        if device == "cpu":
            torch.cuda.empty_cache()
        return self

    # ---------------------------------------------------------------- helpers
    def _split_text_ids(self, response_ids: Sequence[int]):
        """Drop special tokens (EOS etc.) for text alignment, keep index mapping back."""
        special = set(self.policy_tok.all_special_ids or [])
        keep_idx, keep_ids = [], []
        for i, t in enumerate(response_ids):
            if int(t) in special:
                continue
            keep_idx.append(i)
            keep_ids.append(int(t))
        return keep_idx, keep_ids

    # ---------------------------------------------------------------- scoring
    @torch.enable_grad()
    def score(self, problem: str, response_ids: Sequence[int]) -> Dict:
        """Score a single response. `response_ids` must already exclude padding."""
        n = len(response_ids)
        empty = {
            "p2t_token_reward": np.zeros(n, dtype=np.float32),
            "process_token_reward": np.zeros(n, dtype=np.float32),
            "p2t_influence": np.zeros(n, dtype=np.float32),
            "p2t_step_idx": np.full(n, -1, dtype=np.int64),
            "process_info": [],
            "coverage": 0.0,
            "skipped": True,
        }
        if n == 0:
            return empty

        keep_idx, keep_ids = self._split_text_ids(response_ids)
        if not keep_ids:
            return empty
        policy_spans, response = token_char_spans(self.policy_tok, keep_ids)
        if not response.strip():
            return empty

        # keep every split so that "\n\n".join(steps) == response exactly
        steps = response.split("\n\n")
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": self.sep.join(steps) + self.sep},
        ]
        conversation = self.prm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prm_ids = self.prm_tok.encode(conversation, add_special_tokens=False)
        if self.max_prm_len and len(prm_ids) > self.max_prm_len:
            empty["skipped"] = True
            return empty

        prm_spans, _ = token_char_spans(self.prm_tok, prm_ids)

        # first step starts at the assistant content, so prompt tokens are excluded
        content_start = conversation.rfind(self.sep.join(steps) + self.sep)
        response_start = next(
            (j for j, (a, b) in enumerate(prm_spans) if b > content_start), 0
        )

        input_ids = torch.tensor([prm_ids], dtype=torch.long, device=self.device)
        res = compute_p2t_token_rewards(
            self.prm, input_ids, self.sep_id, self.pad_id, self.cfg,
            response_start=response_start,
        )
        if not res.step_rewards:
            return empty

        aligned = align_prm_influence_to_policy(
            steps=steps,
            conversation=conversation,
            sep=self.sep,
            prm_spans=prm_spans,
            prm_influence=res.influence.detach().cpu().numpy(),
            step_rewards=res.step_rewards,
            policy_spans=policy_spans,
            temperature=self.cfg.temperature,
            process_reward_scale=self.process_reward_scale,
        )

        # scatter back onto the original response positions (special tokens keep 0)
        p2t = np.zeros(n, dtype=np.float32)
        proc = np.zeros(n, dtype=np.float32)
        infl = np.zeros(n, dtype=np.float32)
        step_idx = np.full(n, -1, dtype=np.int64)
        for j, i in enumerate(keep_idx):
            p2t[i] = aligned.policy_p2t_reward[j]
            proc[i] = aligned.policy_process_reward[j]
            infl[i] = aligned.policy_influence[j]
            step_idx[i] = aligned.policy_step_idx[j]

        process_info = [
            {
                "process_text": steps[k],
                "process_reward": float(res.step_rewards[k]),
                "process_logits": res.step_logits[k],
            }
            for k in range(min(len(steps), len(res.step_rewards)))
        ]

        return {
            "p2t_token_reward": p2t,
            "process_token_reward": proc,
            "p2t_influence": infl,
            "p2t_step_idx": step_idx,
            "process_info": process_info,
            "coverage": aligned.n_covered / max(aligned.n_total, 1),
            "skipped": False,
            "warnings": aligned.warnings,
        }

    def score_batch(self, problems: Sequence[str], responses_ids: Sequence[Sequence[int]]) -> List[Dict]:
        out = []
        for prob, ids in zip(problems, responses_ids):
            out.append(self.score(prob, ids))
            torch.cuda.empty_cache()
        return out
