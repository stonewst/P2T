# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# Qwen2.5-7B-Instruct has NO thinking mode, so enable_thinking / apply_chat_template_kwargs
# are intentionally omitted (passing enable_thinking would break the Qwen2.5 chat template).

set -x

PROJECT_NAME='P2T'
EXP_NAME='qwen2.5-7b-instruct_MATH-12k'

MODEL_PATH=/home/model/Qwen/Qwen2.5-1.5B-Instruct

data_train_path=data/MATH-12K/train.parquet

prm_path=/home/model/PRM/ReasonFlux-PRM-7B
p2t_sep='<extra_0>'      
p2t_max_len=0           
p2t_offload=cpu         
p2t_grad_ckpt=False      
p2t_attn=flash_attention_2            

p2t_attribution_target=logit_diff  
p2t_backward_mode=single           
p2t_temperature=1.0                 
p2t_process_reward_scale=signed   

p2t_use_outcome_adv=True
p2t_outcome_adv_coef=1.0
p2t_outcome_adv_norm_mode=null   

p2t_use_p2t_adv=True
p2t_p2t_adv_coef=0.1

p2t_use_process_adv=True
p2t_process_adv_coef=0.1


### train
max_prompt_length=2048
max_response_length=8192  
epoch=15
lr=1e-6   
wd=0.01
n_rollout=1
train_temp=1.0   
train_topp=1.0   
use_kl_loss=False
kl_loss_coef=0.0  
kl_loss_type=low_var_kl
entropy_coeff=0.0
train_batchsize=256
ppo_mini_batchsize=64
micro_batchsize_per_gpu=8   
use_dynamic_bsz=False
clip_high=0.28
loss_agg_mode=token-mean
n_gpus_per_node=8
vllm_gpu_memory_util=0.9   


# ------------------------------------------------------------------------
python -m p2t.trainer.main_ppo \
    algorithm.adv_estimator=p2t \
    algorithm.use_kl_in_reward=False \
    algorithm.p2t.prm_path=${prm_path} \
    algorithm.p2t.sep="\"${p2t_sep}\"" \
    algorithm.p2t.max_len=${p2t_max_len} \
    algorithm.p2t.offload=${p2t_offload} \
    algorithm.p2t.grad_checkpointing=${p2t_grad_ckpt} \
    algorithm.p2t.attn_implementation=${p2t_attn} \
    algorithm.p2t.attribution_target=${p2t_attribution_target} \
    algorithm.p2t.backward_mode=${p2t_backward_mode} \
    algorithm.p2t.temperature=${p2t_temperature} \
    algorithm.p2t.process_reward_scale=${p2t_process_reward_scale} \
    algorithm.p2t.use_outcome_adv=${p2t_use_outcome_adv} \
    algorithm.p2t.outcome_adv_coef=${p2t_outcome_adv_coef} \
    algorithm.p2t.outcome_adv_norm_mode=${p2t_outcome_adv_norm_mode} \
    algorithm.p2t.use_p2t_adv=${p2t_use_p2t_adv} \
    algorithm.p2t.p2t_adv_coef=${p2t_p2t_adv_coef} \
    algorithm.p2t.use_process_adv=${p2t_use_process_adv} \
    algorithm.p2t.process_adv_coef=${p2t_process_adv_coef} \
    data.train_files=$data_train_path \
    data.train_batch_size=${train_batchsize} \
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
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_epochs=${epoch} \
    trainer.log_dir='workdir_train/log' \
    trainer.rollout_data_dir='workdir_train/rollout_log/training' \
    trainer.default_local_dir='workdir_train' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME $@
