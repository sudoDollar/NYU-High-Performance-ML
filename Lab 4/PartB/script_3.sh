#!/bin/bash

# Define the values for K, numBlock, and numThread
K_values=(1 5 10 50 100)

numBlock=1
numThread=1
# Loop through combinations of parameters
for K in "${K_values[@]}"; do
    echo "Running program with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done

numBlock=1
numThread=256
# Loop through combinations of parameters
for K in "${K_values[@]}"; do
    echo "Running program with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done

numBlock=-1
numThread=256
# Loop through combinations of parameters
for K in "${K_values[@]}"; do
    echo "Running program with K=$K, numBlock=$numBlock, numThread=$numThread"
    ./q3 $K $numBlock $numThread
done