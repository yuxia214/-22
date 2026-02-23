#!/bin/bash
cd /root/autodl-tmp/MERTools-master/MERBench
python -u mult_robust_v4/run_mult_v4.py \
    --model='mult' \
    --metric_name='emo' \
    --feat_type='frm_align' \
    --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-FRA' \
    --text_feature='Baichuan-13B-Base-FRA' \
    --video_feature='clip-vit-large-patch14-FRA' \
    --gpu=0 \
    --batch_size=512 \
    --num_workers=8 \
    --l2=0.0001 \
    --lr=0.0005 \
    --early_stop_patience=15 \
    --lr_patience=5 \
    --lr_factor=0.5 \
    > mult_v4_train.log 2>&1