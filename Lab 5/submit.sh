#!/bin/bash
#
#SBATCH --job-name=lab5_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --output=%x.out

module purge
module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/Lab_5/

python lab5.py