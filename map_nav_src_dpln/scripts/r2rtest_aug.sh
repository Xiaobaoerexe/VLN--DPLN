#!/bin/bash
DATA_ROOT=../datasets
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型和特征配置
features=vitbase
ft_dim=1024
seed=4
MODELS_DIR=${DATA_ROOT}/R2R/exprs_map/pretrain/cmt-h14-mlm.mrc.sap-init.lxmert-aug.speaker/ckpts
# 输出目录
outdir=${DATA_ROOT}/R2R/exprs_map/batch_test_obj

# 基本参数配置
flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ssrl

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 200

      --batch_size 16
      --lr 5e-6
      --iters 50000
      --log_every 200
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

      --progress_reward_weight 2.0
      --success_reward 10.0
      --revisit_penalty 1.0
      --step_penalty 0.2

      --models_dir ${MODELS_DIR}

      --test"

echo "Starting batch model testing..."

# 创建输出目录
mkdir -p ${outdir}

# 启动8个并行进程，每个使用不同的GPU
for gpu_id in {0..7}
do
    echo "Launching process on GPU ${gpu_id}..."

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    TOTAL_GPUS=8 \
    accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu${gpu_id}_config.yaml \
        r2r/batch_test_models.py $flag \
        --tokenizer bert \
        > ${outdir}/gpu${gpu_id}_log.txt 2>&1 &

    # 保存进程ID
    eval "PID_${gpu_id}=$!"

    # 短暂延迟，避免同时启动造成的资源竞争
    sleep 2
done

echo "All 8 GPU processes launched."
echo "Waiting for all processes to complete..."

# 等待所有进程完成
for gpu_id in {0..7}
do
    eval "wait \$PID_${gpu_id}"
    echo "GPU ${gpu_id} process completed."
done

echo "All GPU processes completed!"

# 合并所有GPU的结果
echo "Merging results from all GPUs..."
python - <<EOF
import json
import os
import glob

outdir = "${outdir}"
all_results = []

# 读取所有GPU的结果文件
for gpu_id in range(8):
    result_file = os.path.join(outdir, f'results_gpu{gpu_id}.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            gpu_results = json.load(f)
            all_results.extend(gpu_results)
            print(f"Loaded {len(gpu_results)} results from GPU {gpu_id}")

# 保存合并后的结果
merged_file = os.path.join(outdir, 'all_results_merged.json')
with open(merged_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nTotal models tested: {len(all_results)}")
print(f"Merged results saved to: {merged_file}")

# 生成汇总报告
successful_results = [r for r in all_results if r is not None]
print(f"Successfully tested: {len(successful_results)}/{len(all_results)} models")

if successful_results:
    # 找出最佳模型
    best_model = max(successful_results,
                    key=lambda x: x['results']['val_unseen']['score_summary']['sr']
                    if 'val_unseen' in x['results'] else 0)

    print(f"\nBest model:")
    print(f"  Name: {best_model['model_name']}")
    print(f"  SR: {best_model['results']['val_unseen']['score_summary']['sr']:.3f}")
    print(f"  SPL: {best_model['results']['val_unseen']['score_summary']['spl']:.3f}")
EOF

echo "Batch testing completed!"
