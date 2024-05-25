### Files:
data.py => CIFAR10 Dataset class  
model.py => ResNet-18 model class  
trainer.py => Helper Class to train the model  
params.py => Configuration file to parse command line arguments

cpu_lab2.py => C2 - C4  
gpu_lab2.py => C5 - C7  
q3_lab2.py => Q3

cpu_submit.sh => sbatch file to submit cpu only jobs (to submit cpu_lab2.py) \
gpu_submit.sh => sbatch file to submit jobs which require GPU (to submit gpu_lab2.py) \
q3_submit.sh => sbatch file to submit q3_lab2.py

lab2.sh => Executable file to submit all jobs sequentially  
(GPU jobs require optimal num workers and hence once that value will be available, GPU job will be submitted automatically).

hpml_cpu.out => Output for CPU jobs (C2 - C4)  
hpml_gpu.out => Output for GPU jobs (C5 - C7)  
hpml_q3.out => Output for Q3

requirements.txt => List of python libraries required to run the program  


### Steps to Run the program

1. In all three sbatch files, need to change the path to location where all program files are present.
```
cd /scratch/ar7996/HPML/Assignment_2/

to

cd <path/to/files>

```

2. In all three sbatch files, need to change annaconda env name
```
source activate /scratch/ar7996/HPML/penv

to

source activate <conda environment for running Pytorch and matplotlib codes>
```

3. Run following commands
```
chmod u+x lab2.sh
./lab2.sh
```

### Additional Info
User can pass custom arguments to python codes as required. Default values has already been set to the values given in assignment.

Guide to enter custom arguments:
```
usage: cpu_lab2.py [-h] [--batch-size N] [--num-workers NUM_WORKERS] [--epochs N] [--lr LR] [--device DEVICE] [--optimizer OPTIMIZER] [--momentum N] [--weight-decay N]
                   [--dataset-path DATASET_PATH]

ResNet18 with CIFAR10 profiling

options:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 128)
  --num-workers NUM_WORKERS
                        Number of I/O processes (default: 2)
  --epochs N            Number of epochs to train (default: 5)
  --lr LR               learning rate (default: 0.1)
  --device DEVICE       Device to be used for training(cpu/cuda)
  --optimizer OPTIMIZER
                        Optimizer to be used (default: sgd)
  --momentum N          Momentum Value (default: 0.9)
  --weight-decay N      Weight Decay value (default 5e-4)
  --dataset-path DATASET_PATH
                        Path of CIFAR10 Dataset (default: ./dataset)

```
