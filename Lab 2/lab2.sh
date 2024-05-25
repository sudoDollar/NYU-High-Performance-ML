#!/bin/bash
file_path="optimal_num_workers.txt"
if [ -f "$file_path" ]; then
    rm $file_path
fi

sbatch cpu_submit.sh

while [ ! -f "$file_path" ]; do
    echo "File $file_path not found. Waiting..."
    sleep 900
done

sbatch gpu_submit.sh

sbatch q3_submit.sh