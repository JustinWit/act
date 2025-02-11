#!/bin/bash
set -ex

python imitate_episodes.py \
--task_name real_pick_coke \
--ckpt_dir act_models/$1 \
--policy_class ACT --kl_weight 10 --chunk_size 5 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 \
--num_epochs 12000 --lr 1e-4 \
--seed 0 \
--run_name $1 \
--preload_data \
--no_proprioception \
