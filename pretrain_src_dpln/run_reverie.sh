#!/bin/bash
outdir=../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker-RAM-l14

echo "使用Accelerate启动分布式训练..."

#使用accelerate launch启动训练
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    train_reverie_obj.py \
    --vlnbert cmt \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_obj_pretrain_CLIP_L14.json \
    --output_dir $outdir \
    --use_dual_policy \
    2>&1 | tee -a $outdir/logs/log.txt

echo "训练完成！"