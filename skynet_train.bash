#!/bin/bash
# SBATCH --job-name=act_training
# SBATCH -p kira-lab
# SBATCH -G 2080_ti:1
# SBATCH -c 7
# SBATCH --qos=short

set -ex
USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate aloha
bash train.bash $1
