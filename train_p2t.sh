export HYDRA_FULL_ERROR=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

set -x

PROJECT_NAME='P2T_project'
EXP_NAME='qwen2.5-math-1.5b-base_MATH12K_P2T'


# HuggingFace repo id (auto-downloaded on first run) or a local path, e.g. /path/to/Qwen2.5-Math-1.5B
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B


# ---------------------------------- Data --------------------------------------
data_train_path=data/MATH-12K/train.parquet
data_test_math500="data/MATH-500/test.parquet"
data_test_path="['$data_test_math500']"



# ---------------------------------- P2T --------------------------------------
use_outcome_adv=True
outcome_adv_coef=1.0

use_p2t_adv=True
p2t_adv_coef=1.0

prm_enable=True
# HuggingFace repo id (auto-downloaded on first run) or a local path, e.g. /path/to/ReasonFlux-PRM-7B
# prm_path=Qwen/Qwen2.5-Math-PRM-7B
prm_path=Gen-Verse/ReasonFlux-PRM-7B
prm_use_pack=True          # False: one PRM step per "\n\n" segment
use_p2t=True
p2t_reward_mode=v4
p2t_residual_coef=2.0
use_process_p2t_reward_calib=True   # repeat calibration (requires repeat.enable)



### ----------------------------------- train -----------------------------------
max_prompt_length=1024
max_response_length=2048   # 2048 / 8192
n_rollout=4
train_temp=0.5     # 1.0 / 0.7
train_topp=1.0    # 1.0 / 0.95
train_topk=-1      # 20 / -1

epoch=10
lr=1e-6   # 1e-6 / 5e-7
wd=0.01
use_kl_loss=True  # False / True
kl_loss_coef=0.001   # 0.001 / 0.0001
kl_loss_type=low_var_kl
entropy_coeff=0.0
clip_high=0.28
loss_agg_mode=seq-mean-token-sum-norm   # token-mean / seq-mean-token-sum / seq-mean-token-mean / seq-mean-token-sum-norm
train_batchsize=128
ppo_mini_batchsize=128
micro_batchsize_per_gpu=4   # 7B + colocated PRM: keep small; raise if memory allows
use_dynamic_bsz=False
nnodes=1
n_gpus_per_node=8
vllm_gpu_memory_util=0.9
use_dynamic_data_filter=False

### val
val_batchsize=500
val_n_rollout=1
val_temp=0.0
val_topp=1.0
val_topk=-1
val_do_sample=False

### checkpoint / eval
save_freq=20
test_freq=20
val_before_train=True


# ------------------------------------------------------------------------
python -m p2t.trainer.main_ppo \
    algorithm.adv_estimator=p2t \
    algorithm.dapo.dynamic_data_filter.enable=${use_dynamic_data_filter} \
    algorithm.use_outcome_adv=${use_outcome_adv} \
    algorithm.outcome_adv_coef=${outcome_adv_coef} \
    algorithm.use_p2t_adv=${use_p2t_adv} \
    algorithm.p2t_adv_coef=${p2t_adv_coef} \
    algorithm.p2t.prm.enable=${prm_enable} \
    algorithm.p2t.prm.prm_path=${prm_path} \
    algorithm.p2t.prm.use_pack_short_process=${prm_use_pack} \
    algorithm.p2t.prm.use_p2t_attribution=${use_p2t} \
    algorithm.p2t.p2t_reward_mode=${p2t_reward_mode} \
    algorithm.p2t.p2t_residual_coef=${p2t_residual_coef} \
    algorithm.use_process_p2t_reward_calib=${use_process_p2t_reward_calib} \
    data.train_files=$data_train_path \
    data.val_files="$data_test_path" \
    data.train_batch_size=${train_batchsize} \
    data.val_batch_size=${val_batchsize} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=${wd} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batchsize} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_memory_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.rollout.temperature=${train_temp} \
    actor_rollout_ref.rollout.top_p=${train_topp} \
    actor_rollout_ref.rollout.top_k=${train_topk} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_rollout} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temp} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_topp} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_topk} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_epochs=${epoch} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.save_per_data_info_to_parquet=False \
    trainer.log_dir='workdir_train/log' \
    trainer.rollout_data_dir='workdir_train/rollout_log/training' \
    trainer.validation_data_dir='workdir_train/rollout_log/validation' \
    trainer.default_local_dir='workdir_train' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME $@
