#!/bin/bash
# TODO add option to set query_frequency and controller temporal frequency
set -ex

# VID_NAME=${3:-"default"}  # optional argument for video name suffix

python imitate_episodes.py \
--eval \
--vid_name $3 \
--ckpt_dir act_models/$1 \
--policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 \
--num_epochs 20000 --lr 1e-4 \
--seed 0 \
--ckpt_name $2 \
--run_name eval \
--absolute_actions \
# --no_proprioception \
# --temporal_agg \
# --full_size_img \
