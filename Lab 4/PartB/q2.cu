#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>

using namespace std;

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char **argv) {

    int blockSize;
    int numBlocks;
    int K, N;

    // Read command line argument
    if(argc == 4){
        sscanf(argv[1], "%d", &K);
        sscanf(argv[2], "%d", &numBlocks);
        sscanf(argv[3], "%d", &blockSize);
        N = K * 1000000;
    } else {
        printf("Usage: %s <NumBlocks(INT or '-1' for variable)>  <NumThreads(INT)>\n", argv[0]);
        exit(0);
    }

    size_t size = N*sizeof(float);
    float *x, *y;

    // Allocate input vectors h_A and h_B in host memory
    float* hx = (float*)malloc(size);
    float* hy = (float*)malloc(size);

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        hx[i] = 1.0f;
        hy[i] = 2.0f;
    }

    double start, stop;
    struct timeval time;

    // Allocate vectors in device memory
    cudaMalloc(&x, size);
    cudaMalloc(&y, size);

    // Copy vectors from host memory to device global memory
    cudaMemcpy(x, hx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, size, cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    if(numBlocks == -1) {
        numBlocks = (N + blockSize - 1) / blockSize;
    }

    //GPU Warm Up
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    cudaMemcpy(hy, y, size, cudaMemcpyDeviceToHost);

    cout << "Number of Elements: " << K << endl;
    cout << "Time Elapsed (GPU): " << stop - start << " secs\n" << endl;

    string fileName = "GPU_K_add_";
    fileName += argv[2];
    fileName += "_";
    fileName += argv[3];
    fileName += ".txt";

    ofstream outfile(fileName, ios::out | ios::app);
    if (outfile.is_open()) {
        outfile << K << " " << stop - start << endl; // Write to file
        outfile.close(); // Close file
    } else {
        cout << "Unable to open file for writing." << endl;
        return 1;
    }

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(hy[i]-3.0f));
    cout << "Max error: " << maxError << endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    free(hx);
    free(hy);
    return 0;
}