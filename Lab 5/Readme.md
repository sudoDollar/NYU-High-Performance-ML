# Part-A: Distributed deep learning

## Files:
1. lab5.py => Main file which contains solution to all the coding questions in order.
2. trainer.py => Helper class to train the model using single GPU
3. ParallelTrainer.py => Helper class to train the model using multiple GPUs as specified in its constructor.
4. submit.sh => Script to submit the job on HPC

## Steps to Run
1. Change Annaconda environment path in submit.sh
2. Change the directory in submit.sh where all required files are present.
```
source activate /path/to/environment/
cd /path/to/lab5
```
3. Finally
```
sbatch submit.sh
```
OR
if running in interactive mode with 4 GPUs
```
module load anaconda3/2020.07
source activate /path/to/environment/
cd /path/to/lab5
python lab5.py
```

