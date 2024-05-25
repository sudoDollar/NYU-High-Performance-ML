#!/bin/bash
#
#SBATCH --job-name=lab4_partC
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x.out

cd /scratch/ar7996/HPML/Lab_4/PartC

module load cuda/11.3.1
module load gcc/10.2.0
module load cudnn/8.6.0.163-cuda11

make clean
make

./c1c2

./c3