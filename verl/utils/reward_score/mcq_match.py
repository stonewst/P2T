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
import re


def compute_score(model_output: str, ground_truth: str) -> bool:
    # --- extract answer from response ---
    extracted_answer = parse_answer_boxed(model_output)

    # --- format ---
    gt = format_text(ground_truth)
    pred = format_text(extracted_answer)
    
    # --- verify correctness ---
    acc = verify_func(gt, pred)

    if acc:
        reward = 1.0
    else:
        reward = 0.0

    return {
        "score": reward,
        "correctness_reward": reward,
        "acc": acc,
    }
    


def verify_func(gt, pred):
    if pred == gt:
        return 1.0
    else:
        return 0.0


def parse_answer_boxed(pred_str):
    ## check fail case-1
    if 'boxed' not in pred_str:
        return ""
    ## check fail case-2
    ans = pred_str.split("boxed")
    if len(ans) == 1:
        return ""
    ## check fail case-3
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    ##
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    a = a.strip(' \n').strip(' \n').strip(' \n')
    return a


def format_text(text):
    answer = text_only_keep_letter(text)
    if len(answer) > 1:
        answer = text_only_keep_upper_letter(answer)
    answer = answer.upper()
    return answer

def text_only_keep_letter(text):
    letters_only = re.sub(r'[^a-zA-Z]', '', text)
    return letters_only

def text_only_keep_upper_letter(text):
    upper_letters_only = re.sub(r'[^A-Z]', '', text)
    return upper_letters_only

