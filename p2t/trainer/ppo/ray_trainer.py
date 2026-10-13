# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
import random
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import math

from p2t.utils.data_sampler.dynamic_batch_sampler import DynamicBatchSampler

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from p2t.trainer.config import AlgoConfig
from p2t.trainer.repeat import (
    apply_process_reward_calibration,
    build_repeat_det_dump_fields,
    build_repeat_detection_kwargs,
    build_repeat_t1_array,
    build_repeat_t2_array,
    compute_response_level_flags,
    compute_response_level_flags_metrics,
    gather_repeat_detection,
    launch_repeat_detection_async,
    repeat_records_to_tuples,
)
from p2t.prm.p2t_reward import P2T_REWARD_MODES, p2t_token_reward_batch, validate_p2t_config
from . import core_algos
from .core_algos import AdvantageEstimator, agg_loss
from .separator import separator_token_lut
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def find_latest_epoch_ckpt_path(root_dir):
    try:
        folders = os.listdir(root_dir)
        folders = [folder for folder in folders if "epoch_" in folder]
        folders.sort(key=lambda x: int(x.split("epoch_")[-1].split("__global_step_")[0]))
        path = os.path.join(root_dir, folders[-1])
    except:
        path = None
    return path

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.P2T:
        # P2T: weighted sum of the outcome and P2T advantages.
        adv_components: dict = {}
        advantages, returns = core_algos.compute_p2t_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            process_reward=data.batch.get("process_reward", None),
            process_step_idx=data.batch.get("process_step_idx", None),
            config=config,
            max_length=data.batch["response_mask"].shape[-1],
            components_out=adv_components,
            p2t_reward=data.batch.get("p2t_reward", None),
            p2t_adv_mask=data.batch.get("p2t_adv_mask", None),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        # per-component advantages, for logging
        for _cname, _ctensor in adv_components.items():
            data.batch[_cname] = _ctensor
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }

        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        # PRM scoring worker (keyed by the string "prm")
        self.use_prm = "prm" in self.role_worker_mapping
        self.use_critic = need_critic(self.config)

        # Repeat-detection results of the current step (see _collect_repeat_detection).
        self._repeat_records = None
        self._repeat_flags = None
        # Step-level repeat interventions need repeat detection and the PRM worker.
        _repeat_cfg = self.config.get("repeat", None)
        _repeat_enabled = bool(_repeat_cfg is not None and _repeat_cfg.get("enable", False))
        for _flag in ("use_process_p2t_reward_calib", "use_mask_after_t2"):
            if not bool(self.config.algorithm.get(_flag, False)):
                continue
            assert _repeat_enabled, (
                f"algorithm.{_flag}=True requires repeat.enable=True so detection provides t1/t2."
            )
            assert self.use_prm, (
                f"algorithm.{_flag}=True requires the PRM worker "
                "(algorithm.p2t.prm.enable=True): the step-level calibration / "
                "tail-trim operate on process_reward + process_step_idx."
            )
        # The P2T advantage needs the PRM worker and the attribution pass.
        if self.config.algorithm.adv_estimator == AdvantageEstimator.P2T:
            if bool(self.config.algorithm.get("use_p2t_adv", False)):
                assert self.use_prm, (
                    "algorithm.use_p2t_adv=True requires the PRM worker "
                    "(algorithm.p2t.prm.enable=True): the term is computed from "
                    "process_reward + process_step_idx."
                )
                assert bool(self.config.algorithm.p2t.prm.get("use_p2t_attribution", False)), (
                    "algorithm.use_p2t_adv=True requires "
                    "algorithm.p2t.prm.use_p2t_attribution=True: without the attribution pass "
                    "there is no per-token R_i for the term to normalize."
                )
        # Validate the P2T reward knobs (algorithm.p2t.*) at startup.
        _prm_cfg = self.config.algorithm.p2t.prm
        if _prm_cfg.get("use_p2t_attribution", False):
            _p2t_cfg = self.config.algorithm.p2t
            _mode = _p2t_cfg.get("p2t_reward_mode", "v0")
            try:
                _shape = validate_p2t_config(
                    _p2t_cfg.get("p2t_weighting", "power"),
                    _p2t_cfg.get("p2t_alpha", 1.0),
                    _p2t_cfg.get("p2t_temperature", 1.0),
                    _p2t_cfg.get("p2t_sharpness_target", None),
                    _mode,
                    _p2t_cfg.get("p2t_residual_coef", 1.0),
                    _p2t_cfg.get("p2t_v4_balance", "min"))
            except ValueError as e:
                raise AssertionError(f"algorithm.p2t.*: {e}") from e
            # Warn about knobs that the selected weighting / mode ignores.
            for _w in _shape.warnings:
                print(f"[p2t] WARNING: algorithm.p2t.{_w}")
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from p2t.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # DAPO dynamic sampling
        enable_dynamic_sampling = (
            hasattr(self.config.algorithm, 'dapo') and hasattr(self.config.algorithm.dapo, 'dynamic_data_filter') and self.config.algorithm.dapo.dynamic_data_filter.enable
        )
        
        if enable_dynamic_sampling:
            # DynamicBatchSampler + DataLoader. num_workers must be 0: with prefetching,
            # set_next_batch_size() would affect a later batch instead of the next one.
            gen_batch_size = int(self.config.data.train_batch_size * self.config.data.gen_batch_size_factor)

            # The initial generation batch (and the one after a force stop) is sized directly,
            # without the gen_batch_size_div round-up, so check its divisibility here.
            div = self.config.data.gen_batch_size_div
            if div > 0 and gen_batch_size % div != 0:
                tbs = self.config.data.train_batch_size
                lo = (gen_batch_size // div) * div
                hints = " or ".join(f"{v / tbs:g} (-> {v})" for v in (lo, lo + div) if v > 0)
                raise ValueError(
                    f"data.train_batch_size ({tbs}) * data.gen_batch_size_factor "
                    f"({self.config.data.gen_batch_size_factor}) = {gen_batch_size}, which is not "
                    f"divisible by data.gen_batch_size_div ({div}). The run's first generation "
                    f"batch (and the first one after every force stop) is sized directly, without "
                    f"the round-up, so it would break the divisor. Use a factor of {hints}, "
                    f"or set gen_batch_size_div=1 if the alignment is not actually needed."
                )

            self.dynamic_batch_sampler = DynamicBatchSampler(
                sampler=train_sampler,
                base_batch_size=gen_batch_size,
                drop_last=True,
            )
            
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_sampler=self.dynamic_batch_sampler,
                num_workers=0,
                collate_fn=collate_fn,
            )
            
            print(f"[DAPO] Dynamic sampling enabled. DataLoader num_workers set to 0 (was {num_workers})")

        
        else:
            # Regular StatefulDataLoader
            self.dynamic_batch_sampler = None
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                # Without DAPO filtering the generation batch is the training batch.
                batch_size=self.config.data.train_batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations_by_data(self, reorg_batch, epoch, dump_path):
        """Dump rollout samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        lines = []
        for data_id, info in reorg_batch.items():
            info2 = {
                'data_id': data_id,
                'epoch': epoch,
                'step': self.global_steps
            }
            info2.update(info)
            lines.append(json.dumps(info2, ensure_ascii=False))
        
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # create the PRM scoring worker, colocated on the global pool
        if self.use_prm:
            resource_pool = self.resource_pool_manager.resource_pool_dict[self.resource_pool_manager.mapping["prm"]]
            # copy the config so the tokenizer path can be added without mutating self.config
            prm_cfg = OmegaConf.create(
                OmegaConf.to_container(self.config.algorithm.p2t.prm, resolve=True)
            )
            prm_cfg.policy_tokenizer_path = self.config.actor_rollout_ref.model.path
            prm_cls = RayClassWithInitArgs(self.role_worker_mapping["prm"], config=prm_cfg)
            self.resource_pool_to_cls[resource_pool]["prm"] = prm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.prm_wg = None
        if self.use_prm:
            self.prm_wg = all_wg["prm"]
            self.prm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _extract_prm_questions(self, batch: DataProto) -> np.ndarray:
        """Raw problem text handed to the PRM, one per response.

        Read from `extra_info["question"]`: the problem without chat template or
        instruction. The data preprocessing must store this field.
        """
        bsz = batch.batch["responses"].shape[0]
        extra = batch.non_tensor_batch.get("extra_info", None)
        if extra is None:
            raise RuntimeError(
                "PRM scoring requires per-sample extra_info with a 'question' field, but "
                "the batch has no 'extra_info'. Ensure data preprocessing stores the raw "
                "problem text in extra_info['question']."
            )
        questions = np.empty(bsz, dtype=object)
        for i in range(bsz):
            info = extra[i]
            q = info.get("question", None) if isinstance(info, dict) else None
            if not q:
                raise RuntimeError(
                    f"PRM scoring requires extra_info['question'] for every sample, but it "
                    f"is missing/empty at index {i}. Ensure data preprocessing stores the "
                    f"raw problem text in extra_info['question']."
                )
            questions[i] = q
        return questions

    # ------------------------------------------------------------------
    # Malignant-repeat detection / penalties
    # ------------------------------------------------------------------

    @property
    def _repeat_enabled(self) -> bool:
        cfg = self.config.get("repeat", None)
        return bool(cfg is not None and cfg.get("enable", False))

    @property
    def _repeat_calib_enabled(self) -> bool:
        return bool(self.config.algorithm.get("use_process_p2t_reward_calib", False))

    @property
    def _repeat_mask_trim_enabled(self) -> bool:
        return bool(self.config.algorithm.get("use_mask_after_t2", False))

    def _launch_repeat_detection(self, batch: DataProto):
        """Fire the async repeat-detection tasks on the finalized training batch.

        Called after the batch is final (post DAPO filtering) and before PRM scoring, so the
        positional records match the batch rows and detection overlaps the PRM forward.
        Returns (futures, valid_lengths), or (None, None) when disabled.
        """
        self._repeat_records = None
        self._repeat_flags = None
        if not self._repeat_enabled:
            return None, None
        cfg = self.config.repeat
        return launch_repeat_detection_async(
            responses=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
            tokenizer_path=self.config.actor_rollout_ref.model.path,
            num_workers=cfg.num_workers,
            detection_kwargs=build_repeat_detection_kwargs(cfg),
            collect_stage_hits=cfg.collect_stage_hits,
            char_max_chars=cfg.char_max_chars,
        )

    def _collect_repeat_detection(self, futures, valid_lengths, acc, metrics) -> dict:
        """Gather the detection results and emit the repeat/* metrics.

        The records (t1/t2) and response-level flags are cached on self for later use. The
        batch must not be reordered between the launch and this call.

        Returns the per-sample dump columns.
        """
        if futures is None:
            return {}
        records, n_t1_fallback = gather_repeat_detection(futures)
        self._repeat_records = records
        results = repeat_records_to_tuples(records)
        max_resp_len = self.config.data.max_response_length
        flags = compute_response_level_flags(results, valid_lengths, max_resp_len)
        self._repeat_flags = flags

        metrics.update(compute_response_level_flags_metrics(flags))

        return build_repeat_det_dump_fields(records, flags, valid_lengths, max_resp_len)

    def _apply_repeat_calibration(self, batch: DataProto, valid_lengths) -> dict:
        """Rewrite R_s inside the repeated region and/or trim the tail past t2.

        Runs after PRM scoring and _collect_repeat_detection. Edits:
          * process_reward: R_s -> algorithm.process_reward_calib_value on calibrated steps.
          * response_mask:  a clone with [t2, end) dropped; the untrimmed mask is kept as
            response_mask_full.
          * process_step_calibrated: (bs, resp_len) int8, 1 on tokens of calibrated steps.

        The same edits are applied to p2t_reward (see _apply_calibration_to_p2t_reward).

        Returns a repeat/* + prm/p2t_* metrics dict.
        """
        if self._repeat_records is None or not (
            self._repeat_calib_enabled or self._repeat_mask_trim_enabled
        ):
            return {}
        records = self._repeat_records
        t1_arr = build_repeat_t1_array(records, valid_lengths) if self._repeat_calib_enabled else None
        t2_arr = build_repeat_t2_array(records, valid_lengths) if self._repeat_mask_trim_enabled else None
        calib_value = float(self.config.algorithm.process_reward_calib_value)

        prew_new, mask_new, mask_full, calibrated, metrics = apply_process_reward_calibration(
            process_reward=batch.batch["process_reward"],
            process_step_idx=batch.batch["process_step_idx"],
            response_mask=batch.batch["response_mask"],
            t1_arr=t1_arr,
            t2_arr=t2_arr,
            calib_value=calib_value,
        )
        batch.batch["process_reward"] = prew_new
        # int8 so the dump stores 0/1
        batch.batch["process_step_calibrated"] = calibrated.to(torch.int8)
        batch.batch["response_mask_full"] = mask_full
        batch.batch["response_mask"] = mask_new
        metrics.update(self._apply_calibration_to_p2t_reward(batch))
        return metrics

    def _log_final_process_reward_mean(self, batch: DataProto, metrics) -> None:
        """Log prm/final_process_reward_mean: the mean per-token R_s after repeat
        calibration and tail trimming (prm/process_reward_mean is the raw PRM output).
        """
        device = batch.batch["process_reward"].device
        if "process_skipped" in batch.batch.keys():
            scored = batch.batch["process_skipped"] < 0.5
        else:
            scored = torch.ones(batch.batch["process_reward"].shape[0],
                                dtype=torch.bool, device=device)
        final_valid = batch.batch["response_mask"].bool() & scored.to(device).unsqueeze(-1)
        metrics["prm/final_process_reward_mean"] = (
            float(batch.batch["process_reward"][final_valid].mean())
            if final_valid.any() else 0.0
        )

    def _compute_process_reward_fields(self, batch: DataProto) -> dict:
        """Score the rollouts with the PRM worker group and attach the process fields.

        Adds to the batch:
            process_reward    (bs, resp_len)  step reward R_s broadcast to its tokens
            process_step_idx  (bs, resp_len)  which step each token belongs to (-1 = none)
            process_skipped   (bs,)           1.0 if the PRM never scored this response
            process_info      non-tensor      list[dict] per response
        Plus, only when use_p2t_attribution is on:
            p2t_influence     (bs, resp_len)  the signed attribution score I_i
        Returns a metrics dict.
        """
        responses = batch.batch["responses"]
        questions = self._extract_prm_questions(batch)

        # send only what the PRM needs
        slim = DataProto.from_dict(
            tensors={
                "responses": responses,
                "response_mask": batch.batch["response_mask"],
            },
            non_tensors={"prm_question": questions},
        )
        slim, pad_size = pad_dataproto_to_divisor(slim, self.prm_wg.world_size)
        out = self.prm_wg.compute_process_rewards(slim)
        out = unpad_dataproto(out, pad_size=pad_size)

        device = responses.device
        batch.batch["process_reward"] = out.batch["process_reward"].to(device)
        batch.batch["process_step_idx"] = out.batch["process_step_idx"].to(device)
        # distinguishes "not scored" from "scored 0"
        batch.batch["process_skipped"] = out.batch["process_skipped"].to(device)
        batch.non_tensor_batch["process_info"] = out.non_tensor_batch["process_info"]

        process_info = out.non_tensor_batch["process_info"]
        # Metrics are averaged over scored responses only; the skip rate is logged separately.
        scored = out.batch["process_skipped"] < 0.5
        n_scored = int(scored.sum())
        n_proc_scored = [len(x) for x, s in zip(process_info, scored.tolist()) if s]
        valid = batch.batch["response_mask"].bool() & scored.to(device).unsqueeze(-1)
        metrics = {
            "prm/skip_ratio": float(out.batch["process_skipped"].mean()),
            "prm/coverage": float(out.batch["process_coverage"][scored].mean()) if n_scored else 0.0,
            "prm/n_process_mean": float(np.mean(n_proc_scored)) if n_proc_scored else 0.0,
            "prm/process_reward_mean": float(batch.batch["process_reward"][valid].mean()) if valid.any() else 0.0,
        }
        # P2T token attribution (only when use_p2t_attribution is on)
        if "p2t_influence" in out.batch.keys():
            batch.batch["p2t_influence"] = out.batch["p2t_influence"].to(device)
            if valid.any():
                # scale of I_i (useful when setting p2t_temperature by hand)
                metrics["prm/p2t_influence_absmean"] = float(
                    batch.batch["p2t_influence"][valid].abs().mean())
        # packing metrics over scored responses (n_blocks == n_segments when packing is off)
        if "process_n_segments" in out.batch and n_scored:
            seg = out.batch["process_n_segments"][scored]
            blk = out.batch["process_n_blocks"][scored]
            seg_cnt = seg.sum().clamp(min=1)
            blk_cnt = blk.sum().clamp(min=1)
            metrics.update({
                "prm/pack_n_segments_mean": float(seg.mean()),
                "prm/pack_n_blocks_mean": float(blk.mean()),
                "prm/pack_ratio": float((seg / blk.clamp(min=1)).mean()),
                "prm/pack_tokens_per_segment_mean": float(out.batch["process_seg_tok_sum"][scored].sum() / seg_cnt),
                "prm/pack_tokens_per_block_mean": float(out.batch["process_blk_tok_sum"][scored].sum() / blk_cnt),
            })
        return metrics

    # ------------------------------------------------------------------ P2T token reward
    def _p2t_reward_pass(self, batch: DataProto, *, valid_mask):
        """Compute R_i from I_i and R_s over valid_mask, using the algorithm.p2t.* knobs."""
        cfg = self.config.algorithm.p2t
        reward_np, stats = p2t_token_reward_batch(
            influence=batch.batch["p2t_influence"].float().cpu().numpy(),
            step_idx=batch.batch["process_step_idx"].cpu().numpy(),
            process_reward=batch.batch["process_reward"].float().cpu().numpy(),
            valid_mask=valid_mask.bool().cpu().numpy(),
            mode=cfg.get("p2t_reward_mode", "v0"),
            weighting=cfg.get("p2t_weighting", "power"),
            # may be "auto"
            alpha=cfg.get("p2t_alpha", 1.0),
            temperature=cfg.get("p2t_temperature", 1.0),
            # None = solver off
            sharpness_target=cfg.get("p2t_sharpness_target", None),
            residual_coef=cfg.get("p2t_residual_coef", 1.0),
            v4_balance=cfg.get("p2t_v4_balance", "min"),
            epsilon=cfg.get("p2t_epsilon", 1e-6),
            clip=cfg.get("p2t_clip", 0.0),
        )
        device = batch.batch["process_reward"].device
        return torch.from_numpy(reward_np).to(device), stats

    def _p2t_metrics(self, batch: DataProto, stats: dict, valid) -> dict:
        """prm/p2t_* metrics of the p2t_reward currently on the batch."""
        if not valid.any():
            return {}
        p2t = batch.batch["p2t_reward"]
        n_steps = max(stats["n_steps"], 1.0)
        out = {
            "prm/p2t_reward_mean": float(p2t[valid].mean()),
            # fraction of degenerate steps (uniform weights)
            "prm/p2t_uniform_ratio": stats["n_uniform"] / n_steps,
            # mean sharpness of the weights (1.0 = uniform)
            "prm/p2t_sharpness": stats["sharpness_sum"] / n_steps,
            "prm/p2t_n_clipped": stats["n_clipped"],
            "prm/p2t_n_steps": stats["n_steps"],
        }
        return out

    def _compute_p2t_reward(self, batch: DataProto) -> dict:
        """Turn I_i into the P2T token reward p2t_reward (bs, resp_len).

        The formula depends on algorithm.p2t.p2t_reward_mode (see prm/p2t_reward.py). Uses the
        raw R_s and the untrimmed mask; repeat calibration is applied afterwards by
        _apply_calibration_to_p2t_reward. Computed whenever use_p2t_attribution is on;
        algorithm.use_p2t_adv decides whether it enters the advantage.
        """
        if "p2t_influence" not in batch.batch.keys():
            return {}
        reward, stats = self._p2t_reward_pass(
            batch,
            valid_mask=batch.batch["response_mask"],
        )
        batch.batch["p2t_reward"] = reward
        return self._p2t_metrics(batch, stats, self._p2t_valid_mask(batch))

    def _apply_calibration_to_p2t_reward(self, batch: DataProto) -> dict:
        """Apply the repeat calibration and tail trim to p2t_reward as masks.

          * calibrated steps -> the constant algorithm.p2t_reward_calib_value.
          * trimmed tail [t2, end) -> 0. The remaining tokens keep their values (no
            renormalization).

        The uncalibrated reward is kept as p2t_reward_raw. Only the reward mean metrics are
        recomputed.
        """
        if "p2t_reward" not in batch.batch.keys():
            return {}
        raw = batch.batch["p2t_reward"]
        batch.batch["p2t_reward_raw"] = raw
        out = raw.clone()
        # absent when only the tail trim is enabled
        cal = batch.batch.get("process_step_calibrated", None)
        if cal is not None:
            cal = cal.bool()
            out[cal] = float(self.config.algorithm.p2t_reward_calib_value)
        # trimmed tail -> 0
        out = out * batch.batch["response_mask"].to(out.dtype)
        batch.batch["p2t_reward"] = out

        metrics = {}
        valid = self._p2t_valid_mask(batch)
        if valid.any():
            metrics["prm/p2t_reward_mean"] = float(out[valid].mean())
        valid_raw = batch.batch["response_mask_full"].bool() & self._p2t_scored_mask(batch)
        if valid_raw.any():
            metrics["prm/p2t_raw_reward_mean"] = float(raw[valid_raw].mean())
        return metrics

    def _p2t_scored_mask(self, batch: DataProto):
        """(bs, 1) bool: rows the PRM actually scored."""
        device = batch.batch["process_reward"].device
        bs = batch.batch["process_reward"].shape[0]
        if "process_skipped" in batch.batch.keys():
            scored = batch.batch["process_skipped"] < 0.5
        else:
            scored = torch.ones(bs, dtype=torch.bool, device=device)
        return scored.to(device).unsqueeze(-1)

    def _p2t_valid_mask(self, batch: DataProto):
        """Tokens the prm/p2t_* metrics average over: supervised AND from a scored response."""
        return batch.batch["response_mask"].bool() & self._p2t_scored_mask(batch)

    def _adv_mask_mode(self, key: str) -> Optional[str]:
        """Read an adv-mask-mode config key; None / "null" both mean 'mask nothing'."""
        mode = self.config.algorithm.get(key, None)
        if mode is None or mode == "null":
            return None
        if mode not in ("sep", "eos", "sep-eos"):
            raise ValueError(
                f"algorithm.{key} must be one of null|sep|eos|sep-eos, got {mode!r}"
            )
        return mode

    @property
    def _p2t_adv_mask_mode(self) -> Optional[str]:
        """p2t_adv_mask_mode from config; None / "null" both mean 'mask nothing'."""
        return self._adv_mask_mode("p2t_adv_mask_mode")

    def _separator_lut(self) -> np.ndarray:
        """Per-token-id flags for tokens containing "\\n\\n" (cached)."""
        lut = getattr(self, "_sep_lut_cache", None)
        if lut is None:
            lut = separator_token_lut(self.tokenizer)
            n_sep = int(lut.sum())
            assert n_sep > 0, (
                "an adv mask mode requests masking '\\n\\n' but no vocabulary entry of "
                f"{type(self.tokenizer).__name__} decodes to anything containing a blank "
                "line, so the mask would be a silent no-op"
            )
            print(f"[p2t] adv mask '\\n\\n' LUT: {n_sep} of {lut.size} vocab entries "
                  f"carry a blank line")
            self._sep_lut_cache = lut
        return lut

    def _build_adv_mask(self, responses: torch.Tensor, mode: str) -> torch.Tensor:
        """(bs, L) bool tensor: True where an advantage term should be zeroed.

        `mode` in {sep, eos, sep-eos}. sep -> "\\n\\n" delimiter tokens (vocab LUT);
        eos -> tokenizer.eos_token_id. Used by p2t_adv_mask_mode.
        """
        resp_np = responses.detach().cpu().numpy()
        mask = np.zeros(resp_np.shape, dtype=bool)
        if mode in ("sep", "sep-eos"):
            mask |= self._separator_lut()[resp_np]
        if mode in ("eos", "sep-eos"):
            eos_id = self.tokenizer.eos_token_id
            assert eos_id is not None, "an adv mask mode masks eos but tokenizer.eos_token_id is None"
            mask |= (resp_np == eos_id)
        return torch.from_numpy(mask)

    def _save_checkpoint(self, local_save_root_dir, global_step, epoch, is_step):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(local_save_root_dir, f"global_step_{global_step}") if is_step else os.path.join(local_save_root_dir, f"epoch_{epoch}__global_step_{global_step}")
        print(f"local_global_step_folder: {local_global_step_folder}")

        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{global_step}", "actor")
        )
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )


        # ----- save actor -----
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, global_step, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # ----- save critic -----
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{global_step}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, global_step, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # ----- save dataloader -----
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        
        if self.dynamic_batch_sampler is not None:
            dataloader_state_dict = {
                'type': 'dynamic_batch_sampler',
                'batch_sampler_state': self.dynamic_batch_sampler.state_dict(),
                'global_steps': global_step,
            }
            print(f"[DAPO] Saving DynamicBatchSampler state")
        else:
            dataloader_state_dict = self.train_dataloader.state_dict()
        
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save train_dataset_state
        train_dataset_state_path = os.path.join(local_global_step_folder, "train_dataset_state.pt")
        torch.save(self.train_dataset.custom_state_dict(), train_dataset_state_path)

        # save current_epoch
        current_epoch_path = os.path.join(local_global_step_folder, "current_epoch.txt")
        with open(current_epoch_path, "w") as f:
            f.write(str(epoch))
        
        # ----- latest checkpointed iteration tracker (for atomic usage) -----
        local_latest_checkpointed_iteration = os.path.join(local_save_root_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _save_checkpoint_step(self, global_step, epoch, is_step=True):
        print('==> saving step_checkpoint ...')
        local_save_root_dir = os.path.join(self.config.trainer.default_local_dir, 'step_checkpoints')
        self._save_checkpoint(local_save_root_dir, global_step, epoch, is_step)

    def _save_checkpoint_epoch(self, global_step, epoch, is_step=False):
        print('==> saving epoch_checkpoint ...')
        local_save_root_dir = os.path.join(self.config.trainer.default_local_dir, 'epoch_checkpoints')
        self._save_checkpoint(local_save_root_dir, global_step, epoch, is_step)


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        

        # (1) auto_from_step
        if self.config.trainer.resume_mode == "auto_from_step":   # auto
            checkpoint_folder = os.path.join(self.config.trainer.default_local_dir, 'step_checkpoints')
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

            if global_step_folder is None:
                print("Training from scratch")
                return 0
        
        # (2) auto_from_epoch
        elif self.config.trainer.resume_mode == "auto_from_epoch":   # auto
            checkpoint_folder = os.path.join(self.config.trainer.default_local_dir, 'epoch_checkpoints')
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_epoch_ckpt_path(checkpoint_folder)  # None if no latest
            
            if global_step_folder is None:
                print("Training from scratch")
                return 0

        # (3) resume_path
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "resume ckpt must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)

        else:
            raise NotImplementedError


        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        
        # -------------- load actor & critic model --------------
        if not self.config.trainer.resume_actor_critic_disabled:
            # load actor
            actor_path = os.path.join(global_step_folder, "actor")
            self.actor_rollout_wg.load_checkpoint(
                actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
            # load critic
            if self.use_critic:
                critic_path = os.path.join(global_step_folder, "critic")
                self.critic_wg.load_checkpoint(
                    critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                )

        # -------------- load dataloader --------------
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            
            if isinstance(dataloader_state_dict, dict) and dataloader_state_dict.get('type') == 'dynamic_batch_sampler':
                if self.dynamic_batch_sampler is not None:
                    self.dynamic_batch_sampler.load_state_dict(dataloader_state_dict['batch_sampler_state'])
                    print(f"[DAPO] Loaded DynamicBatchSampler state")
                else:
                    print(f"Warning: Checkpoint has DynamicBatchSampler state but current config doesn't use it")
            else:
                if self.dynamic_batch_sampler is None:
                    self.train_dataloader.load_state_dict(dataloader_state_dict)
                else:
                    print(f"Warning: Checkpoint has StatefulDataLoader state but current config uses DynamicBatchSampler")
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load current_epoch
        current_epoch_ckpt_path = os.path.join(global_step_folder, "current_epoch.txt")
        print(f'=> loading current_epoch from {current_epoch_ckpt_path}')
        if os.path.exists(current_epoch_ckpt_path):
            with open(current_epoch_ckpt_path, "r") as f:
                ckpt_epoch = int(f.read()) 
            if self.config.trainer.resume_ckpt_is_last_step_in_that_epoch:
                self.current_epoch = ckpt_epoch + 1    # next epoch starts fresh
                print(f'==> set self.current_epoch to {self.current_epoch} (has +1)')
            else:
                self.current_epoch = ckpt_epoch    # resume within the same epoch
                print(f'==> set self.current_epoch to {self.current_epoch}  (no +1)')
        else:
            print(f"Warning: No current_epoch found at {current_epoch_ckpt_path}, will set self.current_epoch to 0 !!!")

        # load train_dataset_state
        train_dataset_state_ckpt_path = os.path.join(global_step_folder, "train_dataset_state.pt")
        print(f'=> loading train_dataset_state from {train_dataset_state_ckpt_path}')
        if os.path.exists(train_dataset_state_ckpt_path):
            train_dataset_state = torch.load(train_dataset_state_ckpt_path, weights_only=False)
            self.train_dataset.load_custom_state_dict(train_dataset_state)
        else:
            print(f"Warning: No train_dataset_custom_state found at {train_dataset_state_ckpt_path}, will start from scratch")


    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    def _build_reward_dict(self, batch: DataProto, reward_extra_infos_dict) -> dict:
        """Collect the reward-like tensors to dump per data item (see _reorganize_info_by_data).

        Each key becomes a `<key>_list` column in the per_data_info parquet, one entry per
        rollout: a per-valid-token list for (bs, resp_len) tensors, a scalar for (bs,) ones.

          total_reward      token_level_rewards (outcome reward, plus the in-reward KL when
                            algorithm.use_kl_in_reward is set).
          outcome_reward    one scalar per rollout, summed from token_level_scores.
          process_reward    PRM step reward R_s broadcast over each step's tokens, after
                            repeat calibration.
          process_step_idx  step id of each token (-1 = none).
          process_step_calibrated
                            1 on tokens of steps overwritten by repeat calibration.
          process_skipped   1.0 when the PRM did not score the rollout.
          p2t_reward        P2T token reward R_i, after repeat calibration.
          p2t_reward_raw    P2T token reward before repeat calibration (only when a repeat
                            intervention ran).
          p2t_influence     signed attribution score I_i.
          *_reward          "_reward"-suffixed entries of reward_extra_infos_dict.
        """
        reward_dict = {"total_reward": batch.batch["token_level_rewards"]}
        reward_dict["outcome_reward"] = batch.batch["token_level_scores"].sum(-1)   # (bs,)

        for key in ("process_reward", "process_step_idx", "process_step_calibrated",
                    "p2t_reward", "p2t_reward_raw", "p2t_influence"):
            if key in batch.batch.keys():
                reward_dict[key] = batch.batch[key]   # (bs, response_length)

        if "process_skipped" in batch.batch.keys():
            reward_dict["process_skipped"] = batch.batch["process_skipped"]   # (bs,)

        # Read the values from the batch: with DAPO dynamic sampling, reward_extra_infos_dict
        # only holds the last generation round.
        for key in (reward_extra_infos_dict or {}):
            if not key.endswith("_reward") or key not in batch.non_tensor_batch:
                continue
            reward_dict[key] = torch.as_tensor(
                np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
            )

        return reward_dict

    def _reorganize_info_by_data(
        self, 
        epoch, step,
        batch,
        reward_dict,   # [key: reward_name]  [value: reward_tensor (bs, response_length)]
        response_mask,
        reward_extra_infos_dict
    ):
        indices = batch.non_tensor_batch["index"].tolist()
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
        rewards = batch.batch["token_level_rewards"].sum(-1).cpu().tolist()

        assert len(indices) == len(inputs) == len(outputs) == len(sample_gts)
        
        reward_extra_infos = [{} for _ in range(len(indices))]
        for k in reward_extra_infos_dict.keys():
            v_list = batch.non_tensor_batch[k].tolist()
            assert len(v_list) == len(indices)
            for i, v in enumerate(v_list):
                reward_extra_infos[i][k] = v
        assert len(reward_extra_infos) == len(indices)


        # collect: input, output, gt, n_rollout, reward_extra_info
        info_per_data = {}
        for _id, _input, _output, _gt, _reward_extra_info, _resp_mask in zip(indices, inputs, outputs, sample_gts, reward_extra_infos, response_mask):
            if _id not in info_per_data:
                info_per_data[_id] = {'input_list': [], 'output_list': [], 'output_length_list': [], 'n_rollout': 0, 'reward_extra_info_list': [], 'gt': None}
            info_per_data[_id]['input_list'].append(_input)
            info_per_data[_id]['output_list'].append(_output)
            resp_len = (_resp_mask==1).float().sum().item()
            info_per_data[_id]['output_length_list'].append(int(resp_len))
            info_per_data[_id]['n_rollout'] += 1
            info_per_data[_id]['gt'] = _gt
            info_per_data[_id]['reward_extra_info_list'].append(_reward_extra_info)
            
        # collect: reward
        for _key, _reward_tensor in reward_dict.items():
            # 1D = response-level, 2D = token-level
            n_dim = _reward_tensor.dim()
            
            for _id, _reward, _resp_mask in zip(indices, _reward_tensor, response_mask):
                _key2 = _key + '_list'

                if _key2 not in info_per_data[_id]:
                    info_per_data[_id][_key2] = []
                
                if n_dim == 1:
                    _reward = _reward.cpu().item()
                elif n_dim == 2:
                    # .bool() is required: an integer mask would index by position.
                    _reward = _reward[_resp_mask.bool()].cpu().tolist()
                
                info_per_data[_id][_key2].append(_reward)
        
        # add: acc_list, mean_acc
        for _id, _info in info_per_data.items():
            acc_list = [item['acc'] for item in _info['reward_extra_info_list']]
            _info['acc_list'] = acc_list
            mean_acc = np.mean(acc_list)
            _info['mean_acc'] = mean_acc
            _info['num_correct_rollout'] = int(mean_acc * _info['n_rollout'])

        return info_per_data, inputs, outputs, sample_gts, rewards


    def _save_info_to_parquet_file(
        self,
        epoch, 
        step,
        info_per_data,
        save_dir,
    ):
        """
        Save token-level rewards to a Parquet file (one row per data item).

        Each row contains:
            - data_id:            data item id
            - epoch:              current epoch
            - step:               current step
            - gt:                 ground-truth
            - input_text:         input text for the data item
            - output_text:        output texts [List]
            - output_length:      actual response lengths [List]
            - acc_list:           per-response accuracy [List]
            - mean_acc:           mean accuracy
            - token-level_reward: nested List, one column per reward type
            - correctness_reward: [List] outcome reward (one value per response; not stored token-level)
        """
        if len(info_per_data) == 0:
            print("Warning: No data to save in _save_info_to_parquet_file")
            return

        
        # iterate over each data item, attach data_id, step, epoch
        rows = []
        for data_id, info_dict in info_per_data.items():
            row = {}
            row['data_id'] = data_id
            row['epoch'] = epoch
            row['step'] = step
            for k, v in info_dict.items():
                if k == 'reward_extra_info':
                    continue
                row[k] = v
            rows.append(row)
        

        # create DataFrame
        df = pd.DataFrame(rows)
        
        # downcast dtypes to save space
        df['data_id'] = df['data_id'].astype('int32')
        df['epoch'] = df['epoch'].astype('int16')
        df['step'] = df['step'].astype('int32')
        df['n_rollout'] = df['n_rollout'].astype('int16')
        df['mean_acc'] = df['mean_acc'].astype('float32')
        df['num_correct_rollout'] = df['num_correct_rollout'].astype('int16')
        
        # create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # save as Parquet with high compression
        filename = f"step_{step}.parquet"
        file_path = save_path / filename
        
        df.to_parquet(
            file_path,
            compression='zstd',  # zstd compression for high ratio
            index=False,
            engine='pyarrow'
        )

        return file_path
        
    def dapo_set_next_gen_batch_size(self, batch_size, gen_bsz_ratio, divisor):
        """Size of the next DAPO generation batch, a multiple of `divisor`.

        Tries, in order, batch_size * gen_bsz_ratio rounded up and down, then batch_size
        rounded up and down, and finally the largest multiple of `divisor` that the remaining
        samples allow.

        Args:
            batch_size: number of prompts still required.
            gen_bsz_ratio: oversampling ratio.
            divisor: required divisor.

        Returns:
            The batch size, or None if fewer than `divisor` samples remain.
        """
        remaining_samples = self.dynamic_batch_sampler.get_remaining_samples_count()
        
        if remaining_samples < divisor:
            return None

        # (1) with the oversampling ratio
        batch_size_aug = int(batch_size * gen_bsz_ratio)

        batch_size_aug_ceil = int(math.ceil(batch_size_aug / divisor) * divisor)
        if remaining_samples >= batch_size_aug_ceil:
            return batch_size_aug_ceil
        
        batch_size_aug_floor = int(math.floor(batch_size_aug / divisor) * divisor)
        if remaining_samples >= batch_size_aug_floor:
            return batch_size_aug_floor
        
        # (2) without the oversampling ratio
        batch_size_ceil = int(math.ceil(batch_size / divisor) * divisor)
        if remaining_samples >= batch_size_ceil:
            return batch_size_ceil
        
        batch_size_floor = int(math.floor(batch_size / divisor) * divisor)
        if remaining_samples >= batch_size_floor:
            return batch_size_floor
        
        # (3) the largest feasible value
        cur_max_bsz = int(math.floor(remaining_samples / divisor) * divisor)
        return cur_max_bsz
    
    
    def dapo_filter_batch(self, batch, num_data_required):
        """Drop prompts whose rollouts are all correct or all wrong; keep at most
        `num_data_required` of the remaining prompts."""
        metric_name = self.config.algorithm.dapo.dynamic_data_filter.metric
        
        # Group by prompt uid and check if all responses have same metric value
        prompt_uid2metric_vals = defaultdict(list)
        prompt_uid2dataid = {}
        for uid, metric_val, data_id in zip(
            batch.non_tensor_batch["uid"], 
            batch.non_tensor_batch[metric_name],
            batch.non_tensor_batch["index"], 
            strict=True
        ):
            prompt_uid2metric_vals[uid].append(metric_val)
            prompt_uid2dataid[uid] = data_id
        
        # Keep prompts with mixed outcomes, or with a single rollout.
        valid_data_uids = []
        invalid_data_uids = []
        invalid_all_correct_uids = []
        invalid_all_wrong_uids = []
        for uid, val_list in prompt_uid2metric_vals.items():
            if len(val_list) == 1:
                valid_data_uids.append(uid)
            else:
                if all(item == 0.0 for item in val_list):
                    invalid_all_wrong_uids.append(uid)
                    invalid_data_uids.append(uid)
                elif all(item == 1.0 for item in val_list):
                    invalid_all_correct_uids.append(uid)
                    invalid_data_uids.append(uid)
                else:
                    valid_data_uids.append(uid)
        
        
        # Randomly keep at most num_data_required valid prompts.
        if len(valid_data_uids) > num_data_required:
            kept_data_uids = random.sample(valid_data_uids, num_data_required)
            valid_but_drop_data_uids = [uid for uid in valid_data_uids if uid not in kept_data_uids]
        else:
            kept_data_uids = valid_data_uids
            valid_but_drop_data_uids = []
        
        kept_data_ids = []
        for uid in kept_data_uids:
            kept_data_ids.append(prompt_uid2dataid[uid])
        
        drop_valid_data_ids = []
        for uid in valid_but_drop_data_uids:
            drop_valid_data_ids.append(prompt_uid2dataid[uid])
        
        drop_invalid_data_ids = []
        for uid in invalid_data_uids:
            drop_invalid_data_ids.append(prompt_uid2dataid[uid])

        drop_invalid_all_correct_data_ids = []
        for uid in invalid_all_correct_uids:
            drop_invalid_all_correct_data_ids.append(prompt_uid2dataid[uid])

        drop_invalid_all_wrong_data_ids = []
        for uid in invalid_all_wrong_uids:
            drop_invalid_all_wrong_data_ids.append(prompt_uid2dataid[uid])


        # Filter trajectories
        kept_traj_idxs = []
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            if uid in kept_data_uids:
                kept_traj_idxs.append(idx)
        filtered_batch = batch[kept_traj_idxs]

        return filtered_batch, kept_data_ids, drop_valid_data_ids, drop_invalid_data_ids, drop_invalid_all_correct_data_ids, drop_invalid_all_wrong_data_ids


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.current_epoch = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # skip_rollout
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        epoch_start = self.current_epoch
        print(f'start_epoch = {epoch_start}')

        # DAPO dynamic sampling state
        enable_dynamic_sampling = self.dynamic_batch_sampler is not None
        
        if enable_dynamic_sampling:
            print(f'DAPO Dynamic Sampling ENABLED')
            print(f'   - metric: {self.config.algorithm.dapo.dynamic_data_filter.metric}')
            print(f'   - gen_batch_size: {self.dynamic_batch_sampler.base_batch_size}')

            # accumulators (persist across dataloader iterations)
            valid_batch = None
            all_batch = None
            num_gen_round = 0
            num_data_generated = 0
            num_data_kept = 0
            num_data_dropped = 0    # invalid + valid-but-dropped
            num_data_valid = 0      # kept + valid-but-dropped
            num_data_invalid = 0
            num_data_invalid_all_correct = 0
            num_data_invalid_all_wrong = 0
            num_data_required = self.config.data.train_batch_size

            valid_ratio = 1.0
            gen_bsz_ratio = (1.0 / valid_ratio) * self.config.data.gen_batch_size_factor
            # divisibility is checked in _create_dataloader
            gen_batch_size = int(self.config.data.train_batch_size * gen_bsz_ratio)
            self.dynamic_batch_sampler.set_next_batch_size(gen_batch_size)


        for epoch in range(epoch_start, self.config.trainer.total_epochs):
            print(f'==> epoch: {epoch}    self.train_dataloader len: {len(self.train_dataloader)}')
            
            self.current_epoch = epoch

            # DAPO: drop samples accumulated in the previous epoch (drop_last). gen_bsz_ratio
            # is kept as the running estimate of the filter yield.
            if enable_dynamic_sampling and valid_batch is not None:
                print(f'==> [DAPO] epoch {epoch} start: dropping {num_data_kept} accumulated samples from previous epoch (drop_last)')
                valid_batch = None
                all_batch = None
                num_gen_round = 0
                num_data_generated = 0
                num_data_kept = 0
                num_data_dropped = 0
                num_data_valid = 0
                num_data_invalid = 0
                num_data_invalid_all_correct = 0
                num_data_invalid_all_wrong = 0
                num_data_required = self.config.data.train_batch_size


            for batch_dict in self.train_dataloader:
                if enable_dynamic_sampling:
                    num_gen_round += 1
                
                # With dynamic sampling, metrics and timing accumulate across generation rounds.
                if enable_dynamic_sampling:
                    if num_gen_round == 1:
                        metrics = {}
                        timing_raw = {}
                else:
                    metrics = {}
                    timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                # Process current batch
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                print(f'==> sample {len(batch)} data')
                
                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        
                        # add reward_extra_infos to batch
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    
                        
                    # DAPO dynamic sampling: filter and accumulate
                    if enable_dynamic_sampling:
                        with marked_timer("dapo", timing_raw, color="magenta"):
                            # collect all
                            if all_batch is None:
                                all_batch = batch
                            else:
                                all_batch = DataProto.concat([all_batch, batch])
                            
                            ## filter batch
                            print(f'==> batch size (before filter): {len(batch)}')
                            filtered_batch, kept_data_ids, drop_valid_data_ids, drop_invalid_data_ids, drop_invalid_all_correct_data_ids, drop_invalid_all_wrong_data_ids = self.dapo_filter_batch(batch, num_data_required)
                            print(f'==> batch size (after filter): {len(filtered_batch)}')

                            ## Accumulate valid_batch
                            if valid_batch is None:
                                valid_batch = filtered_batch
                            else:
                                valid_batch = DataProto.concat([valid_batch, filtered_batch])

                            ## return drop_valid_data_ids to the sampler
                            if len(drop_valid_data_ids) > 0:
                                self.dynamic_batch_sampler.add_to_supp_idx_pool(drop_valid_data_ids)
                            

                            ## update counts (in prompts, not rollouts)
                            cur_num_data_generated = len(np.unique(batch.non_tensor_batch["uid"]))
                            num_data_generated += cur_num_data_generated

                            cur_num_data_kept = len(kept_data_ids)
                            num_data_kept += cur_num_data_kept

                            cur_num_data_drop = len(drop_invalid_data_ids) + len(drop_valid_data_ids)
                            num_data_dropped += cur_num_data_drop

                            cur_num_data_valid = cur_num_data_kept + len(drop_valid_data_ids)
                            num_data_valid += cur_num_data_valid
                            cur_num_data_invalid = len(drop_invalid_data_ids)
                            num_data_invalid += cur_num_data_invalid

                            cur_num_data_invalid_all_correct = len(drop_invalid_all_correct_data_ids)
                            num_data_invalid_all_correct += cur_num_data_invalid_all_correct
                            cur_num_data_invalid_all_wrong = len(drop_invalid_all_wrong_data_ids)
                            num_data_invalid_all_wrong += cur_num_data_invalid_all_wrong
                            
                            num_data_required = self.config.data.train_batch_size - num_data_kept


                            ## print
                            print(f'[DAPO] [Dynamic Sampling] Round = {num_gen_round}: '
                                  f'gen_batch_size (current round):  {self.dynamic_batch_sampler._current_batch_size}, '
                                  f'generate {cur_num_data_generated} data, '
                                  f'valid:  {cur_num_data_valid} data, '
                                  f'invalid:  {cur_num_data_invalid} data, '
                                  f'invalid (all_correct):  {cur_num_data_invalid_all_correct} data, '
                                  f'invalid (all_wrong):  {cur_num_data_invalid_all_wrong} data, '
                                  f'kept:  {cur_num_data_kept} data, '
                                  f'drop:  {cur_num_data_drop} data, '
                                  f'drop_but_valid:  {len(drop_valid_data_ids)} data, '
                                  f'accumulated valid {num_data_kept} data, '
                                  f'still require {num_data_required} data, '
                                )

                            if num_data_required == 0:
                                print('==> Reach train_batch_size.')
                                batch = valid_batch
                            
                            elif num_data_required < 0:
                                raise ValueError

                            elif num_data_required > 0:
                                print(f'==> Still Require {num_data_required} data.')
                                should_force_stop = False
                                gen_batch_size = self.dapo_set_next_gen_batch_size(num_data_required, gen_bsz_ratio, self.config.data.gen_batch_size_div)
                                if gen_batch_size is None:   # data pool exhausted
                                    should_force_stop = True
                                
                                # Force stop: drop the accumulated samples (drop_last) and skip the step.
                                if should_force_stop:
                                    print(f'==> Force stop sampling: pool exhausted, dropping {num_data_kept} accumulated samples (drop_last), skipping this step')
                                    valid_batch = None
                                    all_batch = None
                                    num_gen_round = 0
                                    num_data_generated = 0
                                    num_data_kept = 0
                                    num_data_dropped = 0
                                    num_data_valid = 0
                                    num_data_invalid = 0
                                    num_data_invalid_all_correct = 0
                                    num_data_invalid_all_wrong = 0
                                    num_data_required = self.config.data.train_batch_size
                                    # reset the generation batch size
                                    gen_batch_size = int(self.config.data.train_batch_size * self.config.data.gen_batch_size_factor)
                                    self.dynamic_batch_sampler.set_next_batch_size(gen_batch_size)
                                    continue

                                ## continue sampling
                                else:
                                    print('==> Continue sampling')
                                    self.dynamic_batch_sampler.set_next_batch_size(gen_batch_size)
                                    print(f'gen_batch_size (for next round):  {gen_batch_size}')
                                    continue   # next generation round
                            
                            
                            # Log metrics
                            metrics["dapo/num_gen_round"] = num_gen_round
                            metrics["dapo/num_data_generated"] = num_data_generated

                            metrics["dapo/num_data_drop"] = num_data_dropped
                            metrics["dapo/ratio_data_drop"] = num_data_dropped / num_data_generated

                            metrics["dapo/num_data_kept"] = num_data_kept
                            metrics["dapo/ratio_data_kept"] = num_data_kept / num_data_generated

                            metrics["dapo/num_data_valid"] = num_data_valid
                            metrics["dapo/ratio_data_valid"] = num_data_valid / num_data_generated
                            metrics["dapo/num_data_invalid"] = num_data_invalid
                            metrics["dapo/ratio_data_invalid"] = num_data_invalid / num_data_generated

                            metrics["dapo/num_data_invalid_all_correct"] = num_data_invalid_all_correct
                            metrics["dapo/ratio_data_invalid_all_correct"] = num_data_invalid_all_correct / num_data_generated
                            metrics["dapo/num_data_invalid_all_wrong"] = num_data_invalid_all_wrong
                            metrics["dapo/ratio_data_invalid_all_wrong"] = num_data_invalid_all_wrong / num_data_generated

                    
                    # ------------------------------ Launch Repeat Detection ------------------------------
                    # Launched asynchronously on the final batch; the batch must not be
                    # reordered before the gather below.
                    _repeat_futures, _repeat_valid_lengths = None, None
                    if self._repeat_enabled:
                        with marked_timer("launch_repeat_detection", timing_raw, color="cyan"):
                            _repeat_futures, _repeat_valid_lengths = self._launch_repeat_detection(batch)
                    
                    # ------------------------------ PRM inference ------------------------------
                    # Scores the final training batch (after DAPO filtering).
                    if self.use_prm:
                        with marked_timer("prm", timing_raw, color="cyan"):
                            prm_metrics = self._compute_process_reward_fields(batch)
                            metrics.update(prm_metrics)
                            # I_i -> P2T token reward (only when use_p2t_attribution is on)
                            metrics.update(self._compute_p2t_reward(batch))

                    # -------------------------- Gather Repeat Detection Results ---------------------------
                    if _repeat_futures is not None:
                        with marked_timer("gather_repeat_detection", timing_raw, color="cyan"):
                            _repeat_acc = batch.non_tensor_batch.get("acc", None)
                            _repeat_dump = self._collect_repeat_detection(
                                futures=_repeat_futures,
                                valid_lengths=_repeat_valid_lengths,
                                acc=np.asarray(_repeat_acc) if _repeat_acc is not None else None,
                                metrics=metrics,
                            )
                            # add the detection columns to the per-data dump
                            for _dk, _dv in _repeat_dump.items():
                                batch.non_tensor_batch[_dk] = np.array(_dv, dtype=object)
                                reward_extra_infos_dict[_dk] = _dv

                    # ------------------------------ Repeat Calibration ------------------------------
                    # Rewrite R_s in the repeated region and/or trim the tail past t2. Runs
                    # before old_log_prob so all later stages see the trimmed response_mask.
                    if _repeat_futures is not None and (
                        self._repeat_calib_enabled or self._repeat_mask_trim_enabled
                    ):
                        with marked_timer("repeat_calibration", timing_raw, color="cyan"):
                            metrics.update(self._apply_repeat_calibration(
                                batch, _repeat_valid_lengths,
                            ))

                    # ------------------------- final process_reward metric -------------------------
                    if self.use_prm and "process_reward" in batch.batch.keys():
                        self._log_final_process_reward_mean(batch, metrics)

                    # ------------------------------ recompute old_log_probs ------------------------------
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    # ----------------------------- compute reference log_prob ----------------------------
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ------------------------------ compute values ------------------------------
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    

                    # reward-like tensors dumped to per_data_info
                    reward_dict = self._build_reward_dict(batch, reward_extra_infos_dict)

                    response_mask = batch.batch["response_mask"]
                    if enable_dynamic_sampling:
                        metrics['training/train_acc'] = np.mean(all_batch.non_tensor_batch["acc"])
                        metrics['training/train_acc__only-kept'] = np.mean(batch.non_tensor_batch["acc"])
                    else:
                        metrics['training/train_acc'] = np.mean(batch.non_tensor_batch["acc"])


                    # reorganize info by data
                    with marked_timer("reorganize_info_by_data", timing_raw, color="grey"):
                        info_per_data, inputs, outputs, sample_gts, rewards = self._reorganize_info_by_data(
                                epoch = epoch, 
                                step = self.global_steps, 
                                batch = batch,
                                reward_dict = reward_dict,
                                response_mask = response_mask,
                                reward_extra_infos_dict = reward_extra_infos_dict,
                            )


                    # update per-data stats
                    with marked_timer("update_data_stats", timing_raw, color="skyblue"):
                        self.train_dataset.update_data_stats(epoch, info_per_data)


                    # compute advantage
                    with marked_timer("adv", timing_raw, color="orange"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # p2t_adv_mask_mode: mask that zeros the P2T advantage on sep / eos tokens
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.P2T:
                            _mmode = self._p2t_adv_mask_mode
                            if _mmode is not None:
                                batch.batch["p2t_adv_mask"] = self._build_adv_mask(
                                    batch.batch["responses"], _mmode
                                ).to(batch.batch["responses"].device)

                        # compute advantages, executed on the driver process.
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Add the advantage components to info_per_data, then remove the
                        # temporary keys so they are not sent to the workers.
                        _adv_indices = batch.non_tensor_batch["index"].tolist()
                        for _cname in ("outcome_adv", "p2t_adv"):
                            if _cname not in batch.batch.keys():
                                continue
                            _ctensor = batch.batch[_cname]
                            for _id, _row, _rm in zip(_adv_indices, _ctensor, response_mask):
                                _k2 = _cname + "_list"
                                if _k2 not in info_per_data[_id]:
                                    info_per_data[_id][_k2] = []
                                info_per_data[_id][_k2].append(_row[_rm.bool()].cpu().tolist())
                            batch.batch.pop(_cname)
                        for _mkey in ("p2t_adv_mask",):
                            if _mkey in batch.batch.keys():
                                batch.batch.pop(_mkey)
                        if "response_mask_full" in batch.batch.keys():
                            batch.batch.pop("response_mask_full")


                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=rewards,
                                reward_extra_infos_dict=reward_extra_infos_to_dump,
                                dump_path=rollout_data_dir,
                            )
                    
                    # save per-data info to parquet
                    if self.config.trainer.save_per_data_info_to_parquet:
                        with marked_timer("save_info_to_parquet_file", timing_raw, color="orange"):
                            save_dir = os.path.join(self.config.trainer.log_dir, 'per_data_info')
                            self._save_info_to_parquet_file(
                                epoch = epoch, 
                                step = self.global_steps, 
                                info_per_data = info_per_data,
                                save_dir = save_dir
                            )


                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint_step(self.global_steps, epoch)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                
                # DAPO: reset for the next training step
                if enable_dynamic_sampling:
                    # Filter yield: fraction of generated prompts with mixed outcomes
                    # (valid-but-dropped prompts count as valid).
                    valid_ratio = num_data_valid / num_data_generated if num_data_generated > 0 else 1.0
                    valid_ratio = max(valid_ratio, 1e-6)
                    gen_bsz_ratio = (1.0 / valid_ratio) * self.config.data.gen_batch_size_factor
                    print(f'==> valid_ratio: {valid_ratio}, gen_bsz_ratio: {gen_bsz_ratio}')
                    valid_batch = None
                    all_batch = None
                    num_gen_round = 0
                    num_data_generated = 0
                    num_data_kept = 0
                    num_data_dropped = 0
                    num_data_valid = 0
                    num_data_invalid = 0
                    num_data_invalid_all_correct = 0
                    num_data_invalid_all_wrong = 0
                    num_data_required = self.config.data.train_batch_size
                    gen_batch_size = self.dapo_set_next_gen_batch_size(self.config.data.train_batch_size, gen_bsz_ratio, self.config.data.gen_batch_size_div)  
                    print(f'==> gen_batch_size: {gen_batch_size}')
                    if gen_batch_size is None:
                        gen_batch_size = self.config.data.train_batch_size
                    self.dynamic_batch_sampler.set_next_batch_size(gen_batch_size)
                

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
            
            # -------------------------------------- end of epoch --------------------------------------
            # save checkpoint (at the end of each epoch)
            self._save_checkpoint_epoch(self.global_steps - 1, epoch)
            
            logger.log(data=metrics, step=self.global_steps)

        