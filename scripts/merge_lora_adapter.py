import argparse
import glob
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="lora model merger")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model directory"
    )

    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Paths to LoRA adapter directories. Can use glob patterns like '/path/to/global_step_*/actor/lora_adapter'"
    )

    parser.add_argument(
        "--save_root_dir",
        type=str,
        required=True,
        help="Paths to LoRA adapter directories. Can use glob patterns like '/path/to/global_step_*/actor/lora_adapter'"
    )

    args = parser.parse_args()
    return args





def main():
    args = parse_args()

    ### merge base and lora
    base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype="auto",     
            device_map="auto",
            trust_remote_code=True
        )
    model = PeftModel.from_pretrained(
            base_model,
            args.lora_adapter_path,
        )
    merged_model = model.merge_and_unload()

    ### save
    os.makedirs(args.save_root_dir, exist_ok=True)
    merged_model.save_pretrained(args.save_root_dir, safe_serialization=True)
    print('saved to: ', args.save_root_dir)


main()


