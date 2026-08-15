# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "P2TConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class P2TConfig(BaseConfig):
    """Configuration for Process-to-Token (P2T) reward attribution and advantage.

    Attribution (token influence I_i) and token-reward knobs:
        attribution_target (str): "prob" | "logprob" | "logit_diff" (default, avoids saturation).
        backward_mode (str): "single" (default, O(1) one backward) | "per_step" (ablation, S backward).
        temperature (float): softmax temperature over I_i within a step.
        process_reward_scale (str): map the raw PRM step reward R_s (a probability in
            [0,1]) before broadcast/redistribution. "null" keeps [0,1]; "signed" uses
            2*R_s-1 in [-1,1] so a bad step carries negative reward. Both process and
            p2t inherit the mapped R_s.
        The token reward is R_s*softmax(I_i) (pure within-step redistribution). The
        flat baseline R_s is a separate advantage term (process), so there is no
        with_baseline/no_baseline switch: Eq.5's `R_s + omega*R_s*w_i` is recovered
        as process_adv_coef*R_s + p2t_adv_coef*(R_s*w_i).

    Advantage knobs (final adv = weighted sum of three independently toggled terms):
        A = use_outcome*outcome_coef*A_outcome
          + use_p2t*p2t_coef*R_p2t          (R_p2t = R_s*softmax(I), redistribution)
          + use_process*process_coef*R_process  (R_process = R_s, flat baseline)
        use_outcome_adv (bool), outcome_adv_coef (float, default 1.0),
        outcome_adv_norm_mode (str): null (raw reward) | group-mean (subtract mean)
            | group-mean-std (subtract mean, divide by std; == GRPO, default).
        use_p2t_adv (bool), p2t_adv_coef (float, default 0.6).
        use_process_adv (bool), process_adv_coef (float, default 1.0).
        Paper Eq.5 (alpha=1, omega=0.6) == use_outcome+group-mean-std (coef=1),
        use_p2t (coef=alpha*omega=0.6), use_process (coef=alpha=1.0).

    PRM knobs:
        prm_path (str): path to the process reward model.
        sep (str): step separator token used by the PRM (e.g. "<extra_0>").
        max_len (int): optional cap on PRM input length (0 = no cap).
        offload (str): "cpu" (default) keeps the PRM in host memory between calls,
            "none" leaves it resident on the GPU.
        grad_checkpointing (bool): recompute activations during the attribution
            backward; only needed for very long PRM inputs.
        attn_implementation (str): "sdpa" (default) | "flash_attention_2" | "eager".
            eager materialises the O(L^2) attention matrix (~73GB at 3k tokens,
            OOM at 8k); sdpa/flash keep it O(L) with bf16-identical logits.
    """

    attribution_target: str = "logit_diff"
    backward_mode: str = "single"
    temperature: float = 1.0
    process_reward_scale: str = "null"

    use_outcome_adv: bool = True
    outcome_adv_coef: float = 1.0
    outcome_adv_norm_mode: Optional[str] = "group-mean-std"
    use_p2t_adv: bool = True
    p2t_adv_coef: float = 0.6
    use_process_adv: bool = True
    process_adv_coef: float = 1.0

    prm_path: str = ""
    sep: str = "<extra_0>"
    max_len: int = 0
    offload: str = "cpu"
    grad_checkpointing: bool = False
    attn_implementation: str = "sdpa"


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None

    reward_correct: float = 1.0
    reward_wrong: float = 0.0

    enable_customed_adv_for_all_correct: bool = False
    adv_all_correct: float = 1.0
    enable_customed_adv_for_all_wrong: bool = False
    adv_all_wrong: float = -1.0

    # P2T (Process-to-Token) reward config
    p2t: P2TConfig = field(default_factory=P2TConfig)

