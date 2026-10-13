"""Data-parallel Ray worker that turns rollouts into per-token process rewards.

The worker is colocated with the training workers and claims no GPUs of its own. With
offload="cpu" the PRM stays in host memory and is moved onto the GPU only while scoring.

Input  (sharded by Dispatch.DP_COMPUTE_PROTO):
    responses      (bs, resp_len) int64   rollout token ids, right padded
    response_mask  (bs, resp_len)         1 on real response tokens
    prm_question   non-tensor             problem text handed to the PRM

Output (concatenated back on the driver):
    process_reward    (bs, resp_len) float32   step reward R_s broadcast to its tokens
    process_step_idx  (bs, resp_len) int64     owning step, -1 if none
    process_info      non-tensor               list[dict] per response
    process_coverage  (bs,) float32            fraction of tokens the PRM covered
    process_skipped   (bs,) float32            1 if the response was not scored
    process_n_segments (bs,) float32           finest "\n\n" split count
    process_n_blocks  (bs,) float32            PRM steps after packing (== n_segments off)
    process_seg_tok_{sum,min,max} (bs,) f32    per-response token stats over segments (pre-pack)
    process_blk_tok_{sum,min,max} (bs,) f32    per-response token stats over blocks (post-pack)

Only when use_p2t_attribution is on:
    p2t_influence    (bs, resp_len) f32   the signed attribution score I_i

The P2T token reward is built on the driver (`_compute_p2t_reward` in
trainer/ppo/ray_trainer.py), after repeat calibration may have modified R_s.
"""

import logging
import os

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _object_array(items: list) -> np.ndarray:
    """Build a 1-D object array.

    np.array(list_of_lists, dtype=object) would produce a 2-D array when the inner lists
    share a length.
    """
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


class PRMWorker(Worker):
    """Wraps `ProcessRewardScorer` so a batch of rollouts is scored across all ranks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scorer = None
        self._offload = config.get("offload", "cpu")  # cpu | none

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        from p2t.prm.scorer import ProcessRewardScorer

        cfg = self.config
        policy_tokenizer = hf_tokenizer(copy_to_local(cfg.policy_tokenizer_path), trust_remote_code=True)

        # Load directly onto the resting device.
        resting = "cpu" if self._offload == "cpu" else "cuda"
        self.scorer = ProcessRewardScorer(
            prm_path=copy_to_local(cfg.prm_path),
            policy_tokenizer=policy_tokenizer,
            sep=cfg.sep,
            device=resting,
            system_prompt=cfg.get("system_prompt", None),
            max_prm_len=cfg.get("max_len", 0),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            positive_class=cfg.get("positive_class", 1),
            reward_transform=cfg.get("reward_transform", "prob_diff"),
            use_pack_short_process=cfg.get("use_pack_short_process", False),
            pack_min_token_per_process=cfg.get("pack_min_token_per_process", 32),
            pack_max_token_per_process=cfg.get("pack_max_token_per_process", 128),
            pack_algorithm=cfg.get("pack_algorithm", "dp"),
            pack_over_penalty=cfg.get("pack_over_penalty", 1.0),
            use_p2t_attribution=cfg.get("use_p2t_attribution", False),
            p2t_attribution_target=cfg.get("p2t_attribution_target", "logit_diff"),
            p2t_attribution_backward=cfg.get("p2t_attribution_backward", "single"),
            p2t_attribution_grad_ckpt=cfg.get("p2t_attribution_grad_ckpt", True),
        )
        logger.info(f"[PRMWorker rank {self.rank}] PRM ready, resting on {resting}")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_process_rewards(self, data: DataProto) -> DataProto:
        responses = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        questions = data.non_tensor_batch["prm_question"]
        bsz, rlen = responses.shape

        process_reward = torch.zeros((bsz, rlen), dtype=torch.float32)
        step_idx = torch.full((bsz, rlen), -1, dtype=torch.long)
        coverage = torch.zeros(bsz, dtype=torch.float32)
        skipped = torch.zeros(bsz, dtype=torch.float32)
        n_segments = torch.zeros(bsz, dtype=torch.float32)  # finest "\n\n" split count
        n_blocks = torch.zeros(bsz, dtype=torch.float32)    # PRM steps after packing
        # Per-response token-count stats over segments (pre-pack) and blocks (post-pack).
        # Skipped responses keep 0 and are filtered out on the driver.
        seg_tok_sum = torch.zeros(bsz, dtype=torch.float32)
        seg_tok_min = torch.zeros(bsz, dtype=torch.float32)
        seg_tok_max = torch.zeros(bsz, dtype=torch.float32)
        blk_tok_sum = torch.zeros(bsz, dtype=torch.float32)
        blk_tok_min = torch.zeros(bsz, dtype=torch.float32)
        blk_tok_max = torch.zeros(bsz, dtype=torch.float32)
        # Only emitted when attribution is on.
        p2t_influence = torch.zeros((bsz, rlen), dtype=torch.float32)
        process_info = []

        if self._offload == "cpu":
            self.scorer.to_device("cuda")
        n_failed = 0
        try:
            for i in range(bsz):
                n = int(response_mask[i].sum().item())
                if n == 0:
                    process_info.append([])
                    skipped[i] = 1.0
                    continue
                # On failure (e.g. OOM) skip the sample; its tokens get zero process reward.
                try:
                    out = self.scorer.score(str(questions[i]), responses[i, :n].tolist())
                    process_reward[i, :n] = torch.from_numpy(out["process_reward"])
                    step_idx[i, :n] = torch.from_numpy(out["process_step_idx"])
                    process_info.append(out["process_info"])
                    coverage[i] = out["coverage"]
                    skipped[i] = float(out["skipped"])
                    n_segments[i] = float(out.get("n_segments", 0))
                    n_blocks[i] = float(out.get("n_blocks", 0))
                    seg_tok = out.get("seg_tokens", None)
                    if seg_tok is not None and len(seg_tok):
                        seg_tok_sum[i] = float(seg_tok.sum())
                        seg_tok_min[i] = float(seg_tok.min())
                        seg_tok_max[i] = float(seg_tok.max())
                    blk_tok = out.get("blk_tokens", None)
                    if blk_tok is not None and len(blk_tok):
                        blk_tok_sum[i] = float(blk_tok.sum())
                        blk_tok_min[i] = float(blk_tok.min())
                        blk_tok_max[i] = float(blk_tok.max())
                    p2t_influence[i, :n] = torch.from_numpy(out["p2t_influence"])
                except Exception as e:  # noqa: BLE001 - includes torch.cuda.OutOfMemoryError
                    n_failed += 1
                    skipped[i] = 1.0
                    process_info.append([])
                    logger.warning(f"[PRMWorker rank {self.rank}] score failed on sample {i} (len={n}): {e}")
                    torch.cuda.empty_cache()
        finally:
            if self._offload == "cpu":
                self.scorer.to_device("cpu")
        if n_failed:
            logger.warning(f"[PRMWorker rank {self.rank}] {n_failed}/{bsz} samples skipped after errors")

        tensors = {
            "process_reward": process_reward,
            "process_step_idx": step_idx,
            "process_coverage": coverage,
            "process_skipped": skipped,
            "process_n_segments": n_segments,
            "process_n_blocks": n_blocks,
            "process_seg_tok_sum": seg_tok_sum,
            "process_seg_tok_min": seg_tok_min,
            "process_seg_tok_max": seg_tok_max,
            "process_blk_tok_sum": blk_tok_sum,
            "process_blk_tok_min": blk_tok_min,
            "process_blk_tok_max": blk_tok_max,
        }
        # The driver builds the P2T reward only when p2t_influence is present.
        if self.scorer.use_p2t_attribution:
            tensors["p2t_influence"] = p2t_influence
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={"process_info": _object_array(process_info)},
        )
