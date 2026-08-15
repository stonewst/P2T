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
import contextlib

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    """
    Compute math verification score using math_verify library.
    
    Note: Uses contextlib.suppress to handle all exceptions including TimeoutException.
    This is necessary because math_verify's signal-based timeout mechanism can cause
    issues in Ray's multi-process environment, leading to program hangs.
    
    Args:
        model_output: The model's output string
        ground_truth: The ground truth answer string
        timeout_score: Score to return on timeout (not used with suppress, kept for API compatibility)
    
    Returns:
        ret_score: 0.0 or 1.0 indicating incorrect or correct
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    # Use contextlib.suppress to ignore all exceptions (including TimeoutException)
    # This prevents the program from hanging when timeout occurs in Ray environment
    with contextlib.suppress(Exception):
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])

    return ret_score
