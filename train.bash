#!/bin/bash

python imitate_episodes.py \
--task_name real_pick_coke \
--ckpt_dir act_models/$1 \
--policy_class ACT --kl_weight 10 --chunk_size 5 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
--num_epochs 12000 --lr 5e-5 \
--seed 0 \
--run_name $1 \
--no_proprioception \
