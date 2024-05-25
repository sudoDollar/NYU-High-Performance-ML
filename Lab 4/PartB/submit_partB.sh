#!/bin/bash
#
#SBATCH --job-name=lab4_partB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x.out

cd /scratch/ar7996/HPML/Lab_4/PartB

module load cuda/11.3.1
module load gcc/10.2.0

g++ q1.cpp -o q1

make clean
make

# Define the values for K
K_values=(1 5 10 50 100)

numBlock=1
numThread=1
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q1 program with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q1 $K
done

# Define the values for K, numBlock, and numThread
K_values=(1 5 10 50 100)

numBlock=1
numThread=1
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q2 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q2 $K $numBlock $numThread
done

numBlock=1
numThread=256
# Loop through K values
for K in "${K_values[@]}"; do
    echo "Running Q2 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q2 $K $numBlock $numThread
done

numBlock=-1
numThread=256
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q2 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q2 $K $numBlock $numThread
done


# Define the values for K, numBlock, and numThread
K_values=(1 5 10 50 100)

numBlock=1
numThread=1
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q3 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done

numBlock=1
numThread=256
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q3 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done

numBlock=-1
numThread=256
# Loop through K Values
for K in "${K_values[@]}"; do
    echo "Running Q3 with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done

module load anaconda3/2020.07
source activate /scratch/ar7996/HPML/penv
cd /scratch/ar7996/HPML/Lab_4/PartB

python q4.py