#!/bin/bash
set -ex

python imitate_episodes.py \
--task_name $1 \
--ckpt_dir act_models/$2 \
--policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 \
--num_epochs 810 --lr 1e-3 \
--seed 0 \
--run_name $2 \
--absolute_actions \
--full_size_img \
--gripper_proprio \
--bfloat16 \
--no_wandb \
# --no_proprioception \
# --real_ratio 0.1 \
# --real_data_dir real_pick_coke \
# --preload_to_gpu \
