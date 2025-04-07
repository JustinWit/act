#!/bin/bash
set -ex

python imitate_episodes.py \
--task_name $1 \
--ckpt_dir act_models/$2 \
--policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 50 --dim_feedforward 3200 \
--num_epochs 10000 --lr 5e-5 \
--seed 0 \
--run_name $2 \
--absolute_actions \
--full_size_img \
--no_proprioception \
--real_data_dir real_pick_coke \
--real_ratio 0.1 \
# --preload_to_gpu \
# --preload_to_cpu \
# --gripper_proprio \
# --no_proprioception \
