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
"""
Generate responses given a dataset of prompts
"""

import json
import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)

    # ------------------------------------------------------------------ #
    # Checkpoint path: same directory as output, suffix _ckpt.json        #
    # ------------------------------------------------------------------ #
    output_path = config.data.output_path
    ckpt_path = (
        output_path[: -len(".parquet")] + "_ckpt.json"
        if output_path.endswith(".parquet")
        else output_path + "_ckpt.json"
    )

    # ------------------------------------------------------------------ #
    # Resume: load checkpoint if it exists                                 #
    # ------------------------------------------------------------------ #
    start_batch = 0
    output_lst = [[] for _ in range(config.data.n_samples)]

    if os.path.exists(ckpt_path):
        print(f"[Resume] Checkpoint found at {ckpt_path}, loading ...")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        assert len(ckpt["output_lst"]) == config.data.n_samples, (
            f"Checkpoint n_samples mismatch: checkpoint has {len(ckpt['output_lst'])}, "
            f"but config.data.n_samples={config.data.n_samples}. "
            "Delete the checkpoint file if you want to restart from scratch."
        )
        start_batch = ckpt["completed_batches"]
        output_lst = ckpt["output_lst"]
        already_done = min(start_batch * config_batch_size, total_samples)
        print(
            f"[Resume] Resuming from batch {start_batch + 1}/{num_batch} "
            f"({already_done}/{total_samples} samples already generated)."
        )
    else:
        print("[Info] No checkpoint found, starting from scratch.")

    # ------------------------------------------------------------------ #
    # Main generation loop                                                 #
    # ------------------------------------------------------------------ #
    for batch_idx in range(start_batch, num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)

        # -------------------------------------------------------------- #
        # Save checkpoint after every batch                               #
        # -------------------------------------------------------------- #
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"completed_batches": batch_idx + 1, "output_lst": output_lst}, f)
        done_samples = min((batch_idx + 1) * config_batch_size, total_samples)
        remaining_samples = total_samples - done_samples
        print(
            f"[{batch_idx + 1}/{num_batch}] Checkpoint saved. "
            f"Progress: {done_samples}/{total_samples} samples done, {remaining_samples} remaining."
        )

    # ------------------------------------------------------------------ #
    # Final save                                                           #
    # ------------------------------------------------------------------ #
    # convert output_lst from (n_samples, n_data) to (n_data, n_samples)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)
    print(f"[Done] All {total_samples} samples saved to {config.data.output_path}.")

    # Clean up checkpoint after successful final save
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[Done] Checkpoint {ckpt_path} removed.")


if __name__ == "__main__":
    main()
