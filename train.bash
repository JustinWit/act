#!/bin/bash

python3 imitate_episodes.py \
--task_name real_pick_coke \
--ckpt_dir act_models \
--policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 6000  --lr 2e-5 \
--seed 0
