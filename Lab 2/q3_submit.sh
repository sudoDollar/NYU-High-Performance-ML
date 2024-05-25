#!/bin/bash
#
#SBATCH --job-name=hpml_q3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=6GB
#SBATCH --output=%x.out

module purge
module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/Assignment_2/

python q3_lab2.py