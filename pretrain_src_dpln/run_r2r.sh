#!/bin/bash
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-RAM-l14

echo "使用Accelerate启动分布式训练..."

#使用accelerate launch启动训练
accelerate launch \
    train_r2r.py \
    --vlnbert cmt \
    --model_config config/r2r_model_config_CLIP_L14.json \
    --config config/r2r_pretrain_dual_policy_CLIP_L14.json \
    --output_dir $outdir \
    --use_dual_policy \
    2>&1 | tee -a $outdir/logs/log.txt

echo "训练完成！"