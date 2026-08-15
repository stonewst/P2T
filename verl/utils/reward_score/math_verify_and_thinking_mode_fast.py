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



def compute_score(model_output: str, ground_truth: str) -> bool:
    # correctness
    math_score = compute_score_math_verify(model_output, ground_truth)
    
    # mode
    acc_mode = compute_score_thinking_mode(model_output, gt_mode='no_thinking')

    # reward
    reward = math_score + acc_mode

    return {
        "score": reward,
        "acc_corectness": math_score,
        "acc_mode": acc_mode
    }



def compute_score_thinking_mode(model_output, gt_mode):
    if '</think>' in model_output:
        pred_mode = 'thinking'
    else:
        pred_mode = 'no_thinking'

    if pred_mode == gt_mode:
        return 1.0
    else:
        return 0.0


def compute_score_math_verify(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

