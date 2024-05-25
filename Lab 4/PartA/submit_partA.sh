#!/bin/bash
#
#SBATCH --job-name=lab4_partA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=%x.out

cd /scratch/ar7996/HPML/Lab_4/PartA

module load cuda/11.3.1
module load gcc/10.2.0

make clean
make

# Define the values for number of values per thread
K_values=(500 1000 2000)

# Loop through values
for K in "${K_values[@]}"; do
    echo "Running vecadd00 with number of values per thread=$K"
    ./vecadd00 $K
done

# Define the values for number of values per thread
K_values=(500 1000 2000)

# Loop through values
for K in "${K_values[@]}"; do
    echo "Running vecadd01 with number of values per thread=$K"
    ./vecadd01 $K
done

# Define the values for size of matrices
K_values=(256 512 1024)

# Loop through values
for K in "${K_values[@]}"; do
    echo "Running matmult00 with number of values per thread=$K"
    ./matmult00 $K
done

# Define the values for size of matrices
K_values=(256 512 1024)

# Loop through values
for K in "${K_values[@]}"; do
    echo "Running matmult01 with number of values per thread=$K"
    ./matmult01 $K
done

