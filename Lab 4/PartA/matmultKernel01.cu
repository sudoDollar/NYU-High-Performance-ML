// matmultKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// Multiplies two matrices using CUDA: A x B = C


#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;

  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes four elements of Csub in its copy of CValue
  float Cvalue1 = 0, Cvalue2 = 0, Cvalue3 = 0, Cvalue4 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads FOUR ELEMENTs of ASub and Four of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // // Each thread copies four elements of shared_A and four elements of shared_B
    // BLOCK_SIZE threads will copy BLOCK_SIZE continuos elements at same time in a row.
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_A[thread_row][thread_col+BLOCK_SIZE] = Asub[thread_row * A.stride + thread_col + BLOCK_SIZE];
    shared_A[thread_row + BLOCK_SIZE][thread_col] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];
    shared_A[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col + BLOCK_SIZE];

    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    shared_B[thread_row][thread_col+BLOCK_SIZE] = Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];
    shared_B[thread_row + BLOCK_SIZE][thread_col] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col];
    shared_B[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col + BLOCK_SIZE];
    
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of two rows of shared_A and two cols of shared_B
    // computing Cvalue's by accumulation
#pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e) {
       Cvalue1 += shared_A[thread_row][e] * shared_B[e][thread_col];
       Cvalue2 += shared_A[thread_row][e] * shared_B[e][thread_col + BLOCK_SIZE];
       Cvalue3 += shared_A[thread_row + BLOCK_SIZE][e] * shared_B[e][thread_col];
       Cvalue4 += shared_A[thread_row + BLOCK_SIZE][e] * shared_B[e][thread_col + BLOCK_SIZE];
    }


    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes 4 values it computed.
    Csub[thread_row * C.stride + thread_col] = Cvalue1;
    Csub[thread_row * C.stride + thread_col + BLOCK_SIZE] = Cvalue2;
    Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col] = Cvalue3;
    Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col + BLOCK_SIZE] = Cvalue4;
}

