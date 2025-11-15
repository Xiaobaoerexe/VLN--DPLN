#!/bin/bash
DATA_ROOT=../datasets
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型和特征配置
features=clip-h14
ft_dim=1024
obj_features=timm_vitb16
obj_ft_dim=768
seed=4
MODELS_DIR=${DATA_ROOT}/REVERIE/exprs_map/pretrain/hm3d_rvr/ckpts
# 输出目录
outdir=${DATA_ROOT}/REVERIE/exprs_map/batch_test_obj
# 设置总GPU数量
export TOTAL_GPUS=8

# 对象导航任务的基本参数配置
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

      --train_alg ssrl

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 100
      --max_objects 50

      --batch_size 16
      --lr 5e-6
      --iters 50000
      --log_every 200
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --obj_features ${obj_features}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.6
      --dropout 0.4

      --gamma 0.

      --use_dual_policy

      --models_dir ${MODELS_DIR}

      --test"

echo "Starting batch model testing for Object Navigation..."
echo "Dataset: REVERIE (Object Navigation)"
mkdir -p ${outdir}

# 启动8个并行进程，每个使用不同的GPU
for gpu_id in {0..7}
do
    echo "Launching process on GPU ${gpu_id}..."

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    TOTAL_GPUS=8 \
    accelerate launch --config_file ~/.cache/huggingface/accelerate/gpu${gpu_id}_config.yaml \
        reverie/batch_test_models_obj.py $flag \
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