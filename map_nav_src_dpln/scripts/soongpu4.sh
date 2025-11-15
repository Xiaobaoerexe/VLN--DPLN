DATA_ROOT=../datasets
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=4
train_alg=dsrl
features=vitbase
ft_dim=768
obj_features=butd
obj_ft_dim=2048

seed=4

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-DZRL4

outdir=${DATA_ROOT}/SOON/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset soon
      --output_dir ${outdir}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100

      --batch_size 4
      --lr 5e-6
      --iters 10000
      --log_every 50
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.1

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0.

      --use_dual_policy
      --reward_actor_lr 0.2
      --penalty_actor_lr 0.8
      --use_dynamic_ml_weight 1200
      --use_dynamic_rl_weight 1000

      --lambda_coef 0.4
      --lambda_max 0.9
      --lambda_warmup_steps 1000

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
accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu4_config.yaml soon/main.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../datasets/SOON/exprs_map/pretrain/cmt-vitbase.butdobj-mlm.sap.og-init.lxmert/ckpts/model_step_20000.pt \
      #--eval_first \
      #--aug ../datasets/R2R/annotations/R2R_VLN-RAM_train_enc.json \


# train stage 2
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu4_config.yaml soon/main.py $flag  \
      #--tokenizer bert \
      #--resume_file ../datasets/SOON/trained_models/best_val_unseen_try \
      #--resume_optimizer \
      #--eval_first


# test
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu4_config.yaml soon/main.py $flag  \
      #--tokenizer bert \
      #--resume_file ../datasets/SOON/exprs_map/finetune/dsrl-vitbase-seed.0-DSRL0/ckpts/best_val_unseen \
      #--test --submit
