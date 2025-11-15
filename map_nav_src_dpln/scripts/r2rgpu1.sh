DATA_ROOT=../datasets
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=1
train_alg=dsrl
features=vitbase
ft_dim=768

seed=4

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-DZRL1

outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 16
      --lr 5e-6
      --iters 50000
      --log_every 20
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.6
      --dropout 0.4

      --gamma 0.

      --use_dual_policy
      --reward_actor_lr 0.2
      --penalty_actor_lr 0.8
      --use_dynamic_ml_weight 8000
      --use_dynamic_rl_weight 5000

      --lambda_coef 0.4
      --lambda_max 0.9
      --lambda_warmup_steps 4000

      --memory_size 50

      --num_attention_heads 8
      --fusion_dropout 0.1

      --progress_reward_weight 0.7
      --success_reward 9.0
      --revisit_penalty 1.5
      --step_penalty 0.1
      --failure_penalty_weight 2.0
      --oracle_failure_penalty_weight 2.0
      --early_stop_penalty 3.0"

# train stage 1
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu1_config.yaml r2r/main_nav.py $flag  \
      #--tokenizer bert \
      #--bert_ckpt_file ../datasets/R2R/exprs_map/pretrain/vitbase-dsrl-0626/ckpts/model_step_47500.pt \
      #--eval_first \
      #--aug ../datasets/R2R/annotations/R2R_VLN-RAM_train_enc.json \


# train stage 2
accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu1_config.yaml r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_optimizer \
      --resume_file ../datasets/R2R/trained_models/best_val_unseen_try \
      #--eval_first


# test
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu1_config.yaml r2r/main_nav.py $flag  \
      #--tokenizer bert \
      #--resume_file ../datasets/R2R/exprs_map/finetune/dsrl-vitbase-seed.0-DSRL1/ckpts/best_val_unseen \
      #--test --submit
