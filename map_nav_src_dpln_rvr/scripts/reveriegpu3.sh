DATA_ROOT=../datasets
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=3
train_alg=dsrl
features=clip-h14
ft_dim=1024
obj_features=timm_vitb16
obj_ft_dim=768

seed=4

name=${train_alg}-${features}
name=${name}-seed.${seed}-DSRL_aug3
outdir=${DATA_ROOT}/REVERIE/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
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
      
      --max_action_len 15
      --max_instr_len 100
      --max_objects 50

      --batch_size 16
      --lr 5e-6
      --iters 200000
      --log_every 25
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0.

      --use_dual_policy
      --reward_actor_lr 0.2
      --penalty_actor_lr 0.8
      --use_dynamic_ml_weight 5000
      --use_dynamic_rl_weight 3000

      --lambda_coef 0.4
      --lambda_max 0.9
      --lambda_warmup_steps 2500

      --memory_size 50

      --num_attention_heads 8
      --fusion_dropout 0.1

      --progress_reward_weight 0.8
      --success_reward 10.0
      --revisit_penalty 1.5
      --step_penalty 0.2
      --failure_penalty_weight 3.0
      --oracle_failure_penalty_weight 3.0
      --early_stop_penalty 4.0"

# train stage 1
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu3_config.yaml reverie/main_nav_obj_hm3d.py $flag \
      #--tokenizer bert \
      #--bert_ckpt_file ../datasets/REVERIE/exprs_map/pretrain/hm3d_rvr/ckpts/model_step_194000.pt \
      #--aug ../datasets/REVERIE/annotations/ade20k_pseudo3d_depth2_epoch_94_beam0_sample10.jsonl \
      #--eval_first

# train stage 2
accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu3_config.yaml reverie/main_nav_obj_hm3d.py $flag \
      --tokenizer bert \
      --resume_file ../datasets/REVERIE/trained_models/best_val_aug_52-09_40 \
      #--eval_first

# train stage 3
#accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu3_config.yaml reverie/main_nav_obj_hm3d.py $flag \
      #--tokenizer bert \
      #--resume_file ../datasets/REVERIE/trained_models/best_val_unseen_try \
      #--test --submit