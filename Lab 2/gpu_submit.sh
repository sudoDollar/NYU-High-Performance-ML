#!/bin/bash
#
#SBATCH --job-name=hpml_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x.out

module purge
module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/Assignment_2/

file_path="optimal_num_workers.txt"

if [ -f "$file_path" ]; then
    read -r num_workers < "$file_path"
    python gpu_lab2.py --num-workers $num_workers
fi