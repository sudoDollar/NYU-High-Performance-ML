# Part A

## Problem 1: Vector Addition

### Steps to Run:
```
cd <path/to/PartA>
sbatch submit_partA.sh
```
OR
```
cd <path/to/PartA>
make
// Run each program manually
```


### Description:
In the new kernel vecAddKernel01.cu, all threads access consecutive memory locations in each iteration, making all access coalesce.

It can demostrated as below:
```
T1|T2|T3|.|.|.|.|.|.| => 1st Iteration \
.|.|.|T1|T2|T3|.|.|.| => 2nd Iteration \
.|.|.|.|.|.|T1|T2|T3| => 3rd Iteration and so on...
```

## Problem 2: Matrix Multiplication

### Description:
In new kernel matmultKernel01.cu, each thread access four values as compared to one in original kernel and in each iteration, all threads in same row of a block, access consecutive locations, making it more coalesce.

It can demostrated as below: \
For example, Block Size: 2x2, Footprint Size: 4x4 \
Each block has to access whole footprint elements
```
First Element
|T1|T2|.|.|   |.|.|.|.|
|T3|T4|.|.|   |.|.|.|.|
|.|.|.|.|     |.|.|.|.|
|.|.|.|.|     |.|.|.|.|

|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|

2nd Element
|.|.|T1|T2|   |.|.|.|.|
|.|.|T3|T4|   |.|.|.|.|
|.|.|.|.|     |.|.|.|.|
|.|.|.|.|     |.|.|.|.|

|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|

3rd Element
|.|.|.|.|     |.|.|.|.|
|.|.|.|.|     |.|.|.|.|
|T1|T2|.|.|   |.|.|.|.|
|T3|T4|.|.|   |.|.|.|.|

|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|

4th Element
|.|.|.|.|     |.|.|.|.|
|.|.|.|.|     |.|.|.|.|
|.|.|T1|T2|   |.|.|.|.|
|.|.|T3|T4|   |.|.|.|.|

|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
|.|.|.|.|   |.|.|.|.|
```


# Part B: CUDA Unified Memory

### Steps to Run:
```
cd <path/to/PartB>
sbatch submit_partB.sh
```
OR
```
cd <path/to/PartB>
make
// Run each program manually
```

Profiling details and plots are present in report.

# Part C: Convolution in CUDA

### Steps to Run:
```
cd <path/to/PartC>
sbatch submit_partC.sh
```
OR
```
cd <path/to/PartC>
make
// Run each program manually
```

Execution time and checksum present in report.