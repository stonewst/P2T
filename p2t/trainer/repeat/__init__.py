"""Malignant-repeat detection and penalties for P2T training.

Three layers, each independently switchable:

  detection        ``repeat_detection`` -- per response
                   ``(has_repeat, t_repeat_1, t_repeat_2)`` plus metrics and dump
                   columns. Monitoring only; costs nothing else.
  response-level   a flat -1 on every token of a flagged response, added to the
                   advantage. Unnormalized, so it survives a fully degenerate group.
  step-level       ``repeat_calibration`` -- overwrite the PRM step reward from
                   t_repeat_1 onward, before the step pool is normalized. With
                   ``use_mask_after_t2`` the window narrows to [t1, t2) and everything
                   past t2 leaves both the loss mask and the normalization pool.

The last two are complementary rather than redundant: calibration reshapes rewards
*within* a response and is therefore cancelled by per-response centering when the whole
response is degenerate, which is exactly the case the response-level penalty catches.
"""

from p2t.trainer.repeat.repeat_calibration import (
    STEP_DROPPED,
    apply_process_reward_calibration,
    build_repeat_t1_array,
    build_repeat_t2_array,
    step_is_in_repeat,
    step_supervised_positions,
)
from p2t.trainer.repeat.repeat_detection import (
    REPEAT_DET_SCHEMA_VERSION,
    RepeatDetectRecord,
    build_repeat_det_dump_fields,
    build_repeat_detection_kwargs,
    compute_repeat_monitoring_metrics,
    compute_response_level_flags,
    compute_response_level_flags_metrics,
    compute_truncation_repeat_cross_metrics,
    gather_repeat_detection,
    launch_repeat_detection_async,
    repeat_records_to_tuples,
)

__all__ = [
    "REPEAT_DET_SCHEMA_VERSION",
    "STEP_DROPPED",
    "apply_process_reward_calibration",
    "RepeatDetectRecord",
    "build_repeat_det_dump_fields",
    "build_repeat_detection_kwargs",
    "build_repeat_t1_array",
    "build_repeat_t2_array",
    "compute_repeat_monitoring_metrics",
    "compute_response_level_flags",
    "compute_response_level_flags_metrics",
    "compute_truncation_repeat_cross_metrics",
    "gather_repeat_detection",
    "launch_repeat_detection_async",
    "repeat_records_to_tuples",
    "step_is_in_repeat",
    "step_supervised_positions",
]
