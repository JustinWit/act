#!/bin/bash
# TODO add option to set query_frequency and controller temporal frequency
set -ex

python imitate_episodes.py \
--eval \
--task_name real_pick_coke \
--ckpt_dir act_models/$1 \
--policy_class ACT --kl_weight 10 --chunk_size 5 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 \
# --num_epochs 20000 --lr 1e-4 \
--seed 0 \
--ckpt_name $2 \
--preload_data \
--no_proprioception \
# --temporal_agg \
