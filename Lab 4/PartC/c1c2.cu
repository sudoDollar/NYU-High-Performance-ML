#include<iostream>
#include <sys/time.h>

//Defines for dimensions of tensor
#define height 1024
#define width 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define PADDING 1

typedef struct Tensor{
    int ch;
    int rows;
    int cols;
    double* elements;
} Tensor;

typedef struct Filter{
    int out_ch;
    int ch;
    int rows;
    int cols;
    double* elements;
} Filter;

Tensor get_tensor_host(int ch, int rows, int cols) {
    Tensor tensor;
    tensor.ch = ch;
    tensor.rows = rows;
    tensor.cols = cols;

    //Tensor Allocation on host
    size_t size = tensor.rows * tensor.cols * tensor.ch * sizeof(double);
    tensor.elements = (double*)malloc(size);
    return tensor;
}

void initialize_tensor(Tensor &tensor) {
    //Tensor initialization
    for (int i = 0; i < tensor.ch; ++i) {
        for (int j = 0; j < tensor.rows; ++j) {
            for (int k = 0; k < tensor.cols; ++k) {
                tensor.elements[i * tensor.rows * tensor.cols + j * tensor.cols + k] = (i * (j + k)) * 1.0;
            }
        }
    }
}

Tensor get_padded_tensor(Tensor &I, int pad) {
    int rows = height + 2*pad;
    int cols = width + 2*pad;
    int ch = C;

    Tensor tensor = get_tensor_host(ch, rows, cols);

    for (int i = 0; i < tensor.ch; ++i) {
        for (int j = 0; j < tensor.rows; ++j) {
            for (int k = 0; k < tensor.cols; ++k) {
                if(j < pad || j >= rows - pad || k < pad || k >= cols - pad) {
                    tensor.elements[i * tensor.rows * tensor.cols + j * tensor.cols + k] = 0.0;
                    continue;
                }
                tensor.elements[i * tensor.rows * tensor.cols + j * tensor.cols + k] = I.elements[i * I.rows * I.cols + (j-pad) * I.cols + k-pad];
            }
        }
    }

    return tensor;
}

Filter get_filter_host() {
    Filter filter;
    filter.out_ch = K;
    filter.ch = C;
    filter.rows = FH;
    filter.cols = FW;

    //Tensor Allocation on host
    size_t size = filter.rows * filter.cols * filter.ch * filter.out_ch * sizeof(double);
    filter.elements = (double*)malloc(size);

    //Filter initialization
    for(int i = 0;i<filter.out_ch;i++) {
        for (int j = 0; j < filter.ch; j++) {
            for (int k = 0; k < filter.rows; k++) {
                for (int l = 0; l < filter.cols; l++) {
                    filter.elements[i * filter.ch * filter.rows * filter.cols + j * filter.rows * filter.cols + k * filter.cols + l] = ((i + j) * (k + l)) * 1.0;
                }
            }
        }
    }

    return filter;
}

Tensor get_tensor_device(Tensor &T, bool copy) {
    Tensor tensor;
    tensor.ch = T.ch;
    tensor.rows = T.rows;
    tensor.cols = T.cols;

    size_t size = tensor.rows * tensor.cols * tensor.ch * sizeof(double);

    cudaMalloc((void**) &tensor.elements, size);
    if(copy)
        cudaMemcpy(tensor.elements, T.elements, size, cudaMemcpyHostToDevice);
    return tensor;
}

Filter get_filter_device(Filter &F) {
    Filter filter;
    filter.out_ch = F.out_ch;
    filter.ch = F.ch;
    filter.rows = F.rows;
    filter.cols = F.cols;

    size_t size = filter.rows * filter.cols * filter.ch * filter.out_ch * sizeof(double);

    cudaMalloc((void**) &filter.elements, size);
    cudaMemcpy(filter.elements, F.elements, size, cudaMemcpyHostToDevice);
    return filter;
}

void printTensor(Tensor &T) {
    for(int i=0;i<T.rows;i++) {
        for(int j=0;j<T.cols;j++) {
            std::cout << T.elements[1 * T.rows * T.cols + i * T.cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

double get_filter_element_2(Filter &F, int layer, int ch, int row, int col) {
    return F.elements[layer * F.ch * F.rows * F.cols + ch * F.rows * F.cols + row * F.cols + col];
}

void set_filter_element(Filter &F, int layer, int ch, int row, int col, double val) {
    F.elements[layer * F.ch * F.rows * F.cols + ch * F.rows * F.cols + row * F.cols + col] = val;
}

void transpose(Filter &F) {
    for(int i = 0;i<F.out_ch;i++) {
        for (int j = 0; j < F.ch; j++) {
            for (int k = 0; k < F.rows; k++) {
                for (int l = k+1; l < F.cols; l++) {
                    double tmp = get_filter_element_2(F, i, j, k, l);
                    set_filter_element(F, i, j, k, l, get_filter_element_2(F, i, j, l, k));
                    set_filter_element(F, i, j, l, k, tmp); 
                }
            }
        }
    }
}

 double checksum(Tensor &T) {
    double val = 0.0;

    for (int i = 0; i < T.ch; ++i) {
        for (int j = 0; j < T.rows; ++j) {
            for (int k = 0; k < T.cols; ++k) {
                val += T.elements[i * T.rows * T.cols + j * T.cols + k];
            }
        }
    }

    return val;
}

__device__ void set_element(Tensor &T, int ch, int row, int col, double val) {
    T.elements[ch * T.rows * T.cols + row * T.cols + col] = val;
}

__device__ double get_tensor_element(Tensor &T, int ch, int row, int col) {
    return T.elements[ch * T.rows * T.cols + row * T.cols + col];
}

__device__ double get_filter_element(Filter &F, int layer, int ch, int row, int col) {
    return F.elements[layer * F.ch * F.rows * F.cols + ch * F.rows * F.cols + row * F.cols + col];
}

__device__ double get_conv_element(Tensor &T, Filter &F, int row, int col, int layer) {
    double val = 0.0;

    for(int i=0;i<F.ch;i++) {
        for(int j=0;j<F.rows;j++) {
            for(int k=0;k<F.cols;k++) {
                val += get_filter_element(F, layer, i, F.rows - 1 - j, F.cols - 1 - k) * get_tensor_element(T, i, row + j, col + k);
            }
        }
    }

    return val;
}

__global__ void conv(Tensor I, Tensor O, Filter F) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = blockIdx.z;

    O.elements[layer * O.rows * O.cols + row * O.cols + col] = get_conv_element(I, F, row, col, layer);
}

__global__ void conv_tiling_2(Tensor I, Tensor O, Filter F) {
    int layer = blockIdx.z;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // get the output indices
    int row_o = threadRow + blockIdx.y * TILE_SIZE;
    int col_o = threadCol + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - 1;
    int col_i = col_o - 1;

    //Declare Input tile
    __shared__ double shared_I[C][TILE_SIZE+2][TILE_SIZE+2];
    __shared__ double shared_F[C][FH][FW];

    // Load input tile elements 
    if(row_i >= 0 && row_i < I.rows && col_i >= 0 && col_i < I.cols) {
        shared_I[0][threadRow][threadCol] = get_tensor_element(I, 0, row_i, col_i);
        shared_I[1][threadRow][threadCol] = get_tensor_element(I, 1, row_i, col_i);
        shared_I[2][threadRow][threadCol] = get_tensor_element(I, 2, row_i, col_i);
    }
    else {
        shared_I[0][threadRow][threadCol] = 0.0;
        shared_I[1][threadRow][threadCol] = 0.0;
        shared_I[2][threadRow][threadCol] = 0.0;
    }

    //Load Filter
    if(threadRow < F.rows && threadCol < F.cols) {
        shared_F[0][threadRow][threadCol] = get_filter_element(F, layer, 0, threadRow, threadCol);
        shared_F[1][threadRow][threadCol] = get_filter_element(F, layer, 1, threadRow, threadCol);
        shared_F[2][threadRow][threadCol] = get_filter_element(F, layer, 2, threadRow, threadCol);
    }

    __syncthreads();

    if(threadRow < TILE_SIZE && threadCol < TILE_SIZE){
        double val = 0.0;
        for(int j=0;j<F.rows;j++) {
            for(int k=0;k<F.cols;k++) {
                val += shared_F[0][F.rows-1-j][F.cols-1-k] * shared_I[0][threadRow+j][threadCol+k];
                val += shared_F[1][F.rows-1-j][F.cols-1-k] * shared_I[1][threadRow+j][threadCol+k];
                val += shared_F[2][F.rows-1-j][F.cols-1-k] * shared_I[2][threadRow+j][threadCol+k];
            }
        }
        // only write values if you are inside matrix bounds
        if(row_o < O.rows && col_o < O.cols)
            O.elements[layer * O.rows * O.cols + row_o * O.cols + col_o] = val;
    }
}


int main(void) {

    // Input for C1
    Tensor I = get_tensor_host(C, height, width);
    initialize_tensor(I);
    Tensor I_device = get_tensor_device(I, true);
    // printTensor(I);
    Tensor I_0 = get_padded_tensor(I, 1);
    // printTensor(I_0);
    Filter F = get_filter_host();
    transpose(F);
    Tensor I_0_device = get_tensor_device(I_0, true);
    Filter F_device = get_filter_device(F);

    //Output for C1
    Tensor O = get_tensor_host(K, height, width);
    Tensor O_device = get_tensor_device(O, false);

    // Define grid topology for C1
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE, K);

    double start, stop;
    struct timeval time;

    //Warm Up
    conv<<<dimGrid, dimBlock>>>(I_0_device, O_device, F_device);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");
    start = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    //C1
    conv<<<dimGrid, dimBlock>>>(I_0_device, O_device, F_device);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");
    stop = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    size_t size = O.rows * O.cols * O.ch * sizeof(double);
    cudaMemcpy(O.elements, O_device.elements, size, cudaMemcpyDeviceToHost);
    // printTensor(O);
    //C1 checksum
    double c1_checksum = checksum(O);
    printf("%lf,%.3lf\n", c1_checksum, stop - start);


    // Define grid topology for C2
    dim3 dimBlock2(TILE_SIZE+2, TILE_SIZE+2);
    dim3 dimGrid2(width/TILE_SIZE, height/TILE_SIZE, K);
    
    //GPU WarmUp
    conv_tiling_2<<<dimGrid2, dimBlock2>>>(I_device, O_device, F_device);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");
    start = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    //C2
    conv_tiling_2<<<dimGrid2, dimBlock2>>>(I_device, O_device, F_device);
    cudaDeviceSynchronize();

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");
    stop = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    cudaMemcpy(O.elements, O_device.elements, size, cudaMemcpyDeviceToHost);
    //C2 checksum
    double c2_checksum = checksum(O);
    printf("%lf,%.3lf\n", c2_checksum, stop - start);
    cudaFree(I_0_device.elements);
    cudaFree(O_device.elements);
    cudaFree(F_device.elements);
    cudaFree(I_device.elements);
    free(I.elements);
    free(O.elements);
    free(F.elements);
    free(I_0.elements);

    return 0;
}