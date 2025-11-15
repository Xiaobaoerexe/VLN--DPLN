#!/bin/bash
outdir=../datasets/SOON/exprs_map/pretrain/cmt-vitbase.butdobj-mlm.sap.og-init.lxmert

echo "使用Accelerate启动分布式训练..."

# train
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    train_soon_obj.py \
    --vlnbert cmt \
    --model_config config/soon_obj_model_config.json \
    --config config/soon_obj_pretrain.json \
    --output_dir $outdir \
    --use_dual_policy \
    2>&1 | tee -a $outdir/logs/log.txt

echo "训练完成！"