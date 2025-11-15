#!/bin/bash
outdir=../datasets/REVERIE/exprs_map/pretrain/hm3d_rvr

echo "使用Accelerate启动分布式训练..."

#使用accelerate launch启动训练
accelerate launch --config_file ~/.cache/huggingface/accelerate/2gpu_config.yaml \
    train_hm3d_reverie.py \
    --vlnbert cmt \
    --model_config configs/model_config.json \
    --config configs/training_args.json \
    --output_dir $outdir \
    --use_dual_policy \
    2>&1 | tee -a $outdir/logs/log.txt

echo "训练完成！"
