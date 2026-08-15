"""Data-parallel Ray worker that turns rollouts into P2T token-level rewards.

The worker is colocated with the training workers on the global resource pool, so
it claims no GPUs of its own. The PRM rests in host memory and is moved onto the
GPU only for the duration of a scoring call; leaving it resident would compete
with the rollout engine's KV cache, which is sized as a fraction of the whole card.

Input  (sharded by Dispatch.DP_COMPUTE_PROTO):
    responses      (bs, resp_len) int64   rollout token ids, right padded
    response_mask  (bs, resp_len)         1 on real response tokens
    p2t_question   non-tensor             problem text handed to the PRM

Output (concatenated back on the driver):
    p2t_token_reward     (bs, resp_len) float32   Eq.(3) reward per policy token
    process_token_reward (bs, resp_len) float32   step reward broadcast to its tokens
    p2t_influence        (bs, resp_len) float32   token influence I_i
    p2t_step_idx         (bs, resp_len) int64     owning process, -1 if none
    p2t_process_info     non-tensor               list[dict] per response
    p2t_coverage         (bs,) float32            fraction of tokens the PRM covered
    p2t_skipped          (bs,) float32            1 if the response was not scored
"""

import logging
import os
import sys

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _object_array(items: list) -> np.ndarray:
    """Build a 1-D object array.

    np.array(list_of_lists, dtype=object) collapses into a 2-D array whenever the
    inner lists happen to share a length, which silently corrupts the batch.
    """
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


class P2TWorker(Worker):
    """Wraps `P2TScorer` so a batch of rollouts is scored across all ranks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scorer = None
        self._offload = config.get("offload", "cpu")  # cpu | none

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from p2t_scorer import P2TScorer

        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        cfg = self.config
        policy_tokenizer = hf_tokenizer(copy_to_local(cfg.policy_tokenizer_path), trust_remote_code=True)

        # load straight onto the resting device so a colocated rollout engine never
        # sees a transient 15GB spike at startup
        resting = "cpu" if self._offload == "cpu" else "cuda"
        self.scorer = P2TScorer(
            prm_path=copy_to_local(cfg.prm_path),
            policy_tokenizer=policy_tokenizer,
            sep=cfg.sep,
            device=resting,
            attribution_target=cfg.attribution_target,
            backward_mode=cfg.backward_mode,
            temperature=cfg.temperature,
            process_reward_scale=cfg.get("process_reward_scale", "null"),
            max_prm_len=cfg.max_len,
            grad_checkpointing=cfg.get("grad_checkpointing", False),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
        logger.info(f"[P2TWorker rank {self.rank}] PRM ready, resting on {resting}")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_p2t_rewards(self, data: DataProto) -> DataProto:
        responses = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        questions = data.non_tensor_batch["p2t_question"]
        bsz, rlen = responses.shape

        p2t_reward = torch.zeros((bsz, rlen), dtype=torch.float32)
        process_reward = torch.zeros((bsz, rlen), dtype=torch.float32)
        influence = torch.zeros((bsz, rlen), dtype=torch.float32)
        step_idx = torch.full((bsz, rlen), -1, dtype=torch.long)
        coverage = torch.zeros(bsz, dtype=torch.float32)
        skipped = torch.zeros(bsz, dtype=torch.float32)
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
                # a single pathological response (e.g. a degenerate 8k-token repeat
                # from an untrained policy) must not OOM-hang the whole step: on any
                # failure we skip that sample, its tokens get zero P2T reward and it
                # falls back to the plain outcome advantage.
                try:
                    out = self.scorer.score(str(questions[i]), responses[i, :n].tolist())
                    p2t_reward[i, :n] = torch.from_numpy(out["p2t_token_reward"])
                    process_reward[i, :n] = torch.from_numpy(out["process_token_reward"])
                    influence[i, :n] = torch.from_numpy(out["p2t_influence"])
                    step_idx[i, :n] = torch.from_numpy(out["p2t_step_idx"])
                    process_info.append(out["process_info"])
                    coverage[i] = out["coverage"]
                    skipped[i] = float(out["skipped"])
                except Exception as e:  # noqa: BLE001 - includes torch.cuda.OutOfMemoryError
                    n_failed += 1
                    skipped[i] = 1.0
                    process_info.append([])
                    logger.warning(f"[P2TWorker rank {self.rank}] score failed on sample {i} (len={n}): {e}")
                    torch.cuda.empty_cache()
        finally:
            if self._offload == "cpu":
                self.scorer.to_device("cpu")
        if n_failed:
            logger.warning(f"[P2TWorker rank {self.rank}] {n_failed}/{bsz} samples skipped after errors")

        return DataProto.from_dict(
            tensors={
                "p2t_token_reward": p2t_reward,
                "process_token_reward": process_reward,
                "p2t_influence": influence,
                "p2t_step_idx": step_idx,
                "p2t_coverage": coverage,
                "p2t_skipped": skipped,
            },
            non_tensors={"p2t_process_info": _object_array(process_info)},
        )
