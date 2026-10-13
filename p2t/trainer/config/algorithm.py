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
from typing import Any

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "KLControlConfig", "RepeatConfig"]


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
class RepeatConfig(BaseConfig):
    """Configuration for malignant-repeat DETECTION (detection knobs only).

    Loaded as a top-level `repeat` key (see trainer/config/repeat/repeat.yaml); access
    in code via `config.repeat.<field>`.

    Detection runs `detect_malignant_repeat_pipeline`, a four-level fallback chain
    (word block hashing -> char block hashing -> \\boxed{} spam -> word/concat spam).

    The step-level interventions are configured in AlgoConfig
    (algorithm.use_process_p2t_reward_calib / use_mask_after_t2) and require enable=True.

    Args:
        enable (bool): enable detection and the repeat/* metrics; does not change the loss
            by itself.
        num_workers (int): parallel Ray chunks for detection.
        collect_stage_hits (bool): also record which levels would fire (slow; diagnosis only).
        char_max_chars (int): skip the char-level fallback for texts longer than this
            (0 = no cap).
        block_size / min_repeats / max_consecutive_mismatch / min_period /
        min_unique_tokens_in_unit: word-level detector knobs.
        char_*: char-level fallback knobs.
        boxed_*: \\boxed{} spam knobs.
        spam_*: word/concat spam knobs.
    """

    enable: bool = False
    num_workers: int = 8
    collect_stage_hits: bool = False
    char_max_chars: int = 0

    block_size: int = 12
    min_repeats: int = 3
    max_consecutive_mismatch: int = 3
    min_period: int = 2
    min_unique_tokens_in_unit: int = 2

    char_fallback: bool = True
    char_block_size: int = 50
    char_min_period: int = 15
    char_min_unique: int = 3

    boxed_fallback: bool = True
    boxed_threshold: int = 5
    boxed_dominant_ratio: float = 0.75
    boxed_m_index: int = 4
    boxed_tail_chars: int = 0

    spam_fallback: bool = True
    spam_word_min_count: int = 15
    spam_word_min_ratio: float = 0.35
    spam_word_min_run: int = 10
    spam_concat_min_repeats: int = 15
    spam_concat_min_unit_len: int = 3
    spam_concat_max_unit_len: int = 40


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

    DAPO filtering is configured under `algorithm.dapo.dynamic_data_filter` (see
    p2t_ppo_trainer.yaml).
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

    # P2T advantage (compute_p2t_advantage): total advantage is a weighted
    # sum of an outcome advantage and a P2T token advantage.
    #   use_outcome_adv (bool): include the outcome advantage term.
    #   outcome_adv_coef (float): weight of the outcome advantage.
    #   outcome_adv_norm_mode (str): group-mean | group-mean-std | rloo-mean.
    use_outcome_adv: bool = True
    outcome_adv_coef: float = 1.0
    outcome_adv_norm_mode: str = "group-mean"

    # P2T advantage, built from the per-token P2T reward R_i (see algorithm.p2t.*).
    #   use_p2t_adv (bool): include the term. Requires algorithm.p2t.prm.use_p2t_attribution=True.
    #   p2t_adv_coef (float): weight of the term.
    #   p2t_adv_mode (str): token-level-norm | process-level-norm
    #       token-level-norm: center R_i by a token-level baseline.
    #       process-level-norm: center the step reward (A_s = R_s - step baseline), then
    #         A_i = A_s * R_i / R_s.
    #   p2t_adv_norm_mode (str): group-mean | rloo-mean | response-mean
    #   p2t_adv_token_denom (str): valid | max  (token-level-norm only)
    use_p2t_adv: bool = True
    p2t_adv_coef: float = 1.0
    p2t_adv_mode: str = "token-level-norm"
    p2t_adv_norm_mode: str = "group-mean"
    p2t_adv_token_denom: str = "valid"
    # Zero the P2T advantage on these tokens: null | sep ("\n\n") | eos | sep-eos.
    # Masked tokens still enter the baseline.
    p2t_adv_mask_mode: str = "sep"

    # Step-level repeat interventions (require repeat.enable=True and the PRM worker).
    #   use_process_p2t_reward_calib (bool): set R_s to process_reward_calib_value on steps
    #     overlapping [t1, end), and p2t_reward to p2t_reward_calib_value on those tokens.
    #   process_reward_calib_value (float): value substituted for R_s.
    #   p2t_reward_calib_value (float): value substituted for p2t_reward.
    #   use_mask_after_t2 (bool): narrow calibration to [t1, t2) and drop [t2, end) from the
    #     loss and the normalization pool.
    use_process_p2t_reward_calib: bool = False
    process_reward_calib_value: float = 0.0
    p2t_reward_calib_value: float = 0.0
    use_mask_after_t2: bool = False

