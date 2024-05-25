// This Kernel adds two Vectors A and B in C on GPU
// using coalesced memory access.


__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int totalThreads = gridDim.x * blockDim.x;
    int threadStartIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int threadEndIndex   = totalThreads * (N - 1) + threadStartIndex;
    
    int i;

    //All threads access consecutive locations in each iteration
    //after each iteration, thread moves to 'totalThreads' position ahead 
    for( i=threadStartIndex; i<=threadEndIndex; i+=totalThreads ){
        C[i] = A[i] + B[i];
    }
}