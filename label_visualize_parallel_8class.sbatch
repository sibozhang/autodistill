#!/bin/bash
#SBATCH --partition=V100x8_sixdays,A100x8_twodays,A100x8,V100x8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/autodistill/logs/label_visualize_parallel_8class_%j.out
#SBATCH --error=/mnt/scratch/sibo/autodistill/logs/label_visualize_parallel_8class_%j.err
#SBATCH --exclude=asimov-240,asimov-243

cd /mnt/scratch/sibo/autodistill
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate autodistill

python label_visualize_parallel_8class.py