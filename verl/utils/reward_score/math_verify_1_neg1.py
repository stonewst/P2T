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

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = -1.0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    ### 
    acc = 0.0
    
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        acc, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        pass
    
    if acc:
        reward = 1.0
    else:
        reward = -1.0

    return {
        "score": reward,
        "correctness_reward": reward,
        "acc": acc,
    }