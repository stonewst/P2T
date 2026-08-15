# Copyright 2024 PRIME team and/or its affiliates
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

import json
import traceback

from .utils import check_correctness as apps_check_correctness


def compute_score(completion, test_cases, continuous=False):
    """
    Args:
        completion: full model output (may contain prose + markdown code blocks)
        test_cases: dict or JSON string of test cases
        continuous: whether to return a continuous score (pass rate) instead of binary
    Returns:
        success:
            - Binary mode: True/False
            - Continuous mode: float in [0.0, 1.0] (pass rate)
        metadata_list:
            - continuous mode: list of per-test-case details
            - binary mode: single metadata dict
    """
    ### 1. Extract Python code from completion
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # 2. Run all test cases first (quick check).
        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass
        
        # 3. If there were failures or a continuous score is needed, test case by case.

        # Split test cases into individual dicts.
        # Original format: {"inputs": [in1, in2, in3], "outputs": [out1, out2, out3]}
        # Split format:    [{"inputs": [in1], "outputs": [out1]}, {"inputs": [in2], "outputs": [out2]}, ...]
        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})
            

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []

            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)   
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)  # res is a list; extend to flatten

                if test_case_id >= 9:  # test at most 10 cases (indices 0-9)
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1  # avoid division by zero
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    
    return success, metadata_list
