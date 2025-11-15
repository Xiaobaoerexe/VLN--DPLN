#!/bin/bash
outdir=../datasets/RxR/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-RAM-l14

echo "使用Accelerate启动分布式训练..."

#使用accelerate launch启动训练
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    train_r2r.py \
    --vlnbert cmt \
    --model_config config/rxr_model_config.json \
    --config config/rxr_pretrain.json \
    --output_dir $outdir \
    --use_dual_policy \
    2>&1 | tee -a $outdir/logs/log.txt

echo "训练完成！"