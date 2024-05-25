#!/bin/bash
#
#SBATCH --job-name=hpml_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --mem=6GB
#SBATCH --output=%x.out

module purge
module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/Assignment_2/

python cpu_lab2.py