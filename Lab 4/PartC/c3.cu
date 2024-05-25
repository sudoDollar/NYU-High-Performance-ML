#include<iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <string>

//Defines for dimensions of tensor
#define height 1024
#define width 1024
#define C 3
#define FW 3
#define FH 3
#define K 64

#define checkCUDNN(expression) do { \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cout << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status)<<  std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
} while(0)

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
                    filter.elements[i * filter.ch * filter.rows * filter.cols + j * filter.rows * filter.cols + k * filter.cols + l] = (i + j) * (k + l) * 1.0;
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

double get_filter_element_2(Filter &F, int layer, int ch, int row, int col) {
    return F.elements[layer * F.ch * F.rows * F.cols + ch * F.rows * F.cols + row * F.cols + col];
}

void set_filter_element(Filter &F, int layer, int ch, int row, int col, double val) {
    F.elements[layer * F.ch * F.rows * F.cols + ch * F.rows * F.cols + row * F.cols + col] = val;
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

int main(void) {

    int device = 0; // Select GPU device 0 (change this to select a different GPU)
    cudaSetDevice(device);

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_DOUBLE,
                                        /*batch_size=*/1,
                                        /*channels=*/C,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_DOUBLE,
                                        /*batch_size=*/1,
                                        /*channels=*/K,
                                        /*image_height=*/height,
                                        /*image_width=*/width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_DOUBLE,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/K,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/FH,
                                        /*kernel_width=*/FW));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            /*pad_height=*/1,
                                            /*pad_width=*/1,
                                            /*vertical_stride=*/1,
                                            /*horizontal_stride=*/1,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CONVOLUTION,
                                            /*computeType=*/CUDNN_DATA_DOUBLE));

    // Get the number of convolution algorithms available
    int requestedAlgoCount = 7; // Requesting 5 algorithms
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            requestedAlgoCount,
                                            &returnedAlgoCount,
                                            perfResults));

    // Select the fastest algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm = perfResults[0].algo; // Default to the fastest algorithm

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));


    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    //Input for C3
    Tensor I = get_tensor_host(C, height, width);
    initialize_tensor(I);
    Filter F = get_filter_host();

    Tensor I_device = get_tensor_device(I, true);
    Filter F_device = get_filter_device(F);

    //Output for C3
    Tensor O = get_tensor_host(K, height, width);
    size_t size = O.rows * O.cols * O.ch * sizeof(double);
    Tensor O_device = get_tensor_device(O, false);


    const double alpha = 1.0, beta = 0.0;

    double start, stop;
    struct timeval time;

    //Warm Up

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    I_device.elements,
                                    kernel_descriptor,
                                    F_device.elements,
                                    convolution_descriptor,
                                    convolution_algorithm,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    output_descriptor,
                                    O_device.elements));


    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");
    start = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    I_device.elements,
                                    kernel_descriptor,
                                    F_device.elements,
                                    convolution_descriptor,
                                    convolution_algorithm,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    output_descriptor,
                                    O_device.elements));

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");
    stop = ((double)time.tv_sec * 1000.0) + ((double)time.tv_usec / 1000.0);

    cudaMemcpy(O.elements, O_device.elements, size, cudaMemcpyDeviceToHost);

    //C3 checksum
    double c3_checksum = checksum(O);
    printf("%lf,%.3lf\n", c3_checksum, stop - start);

    // Clean up
    cudaFree(I_device.elements);
    cudaFree(F_device.elements);
    cudaFree(O_device.elements);
    free(I.elements);
    free(O.elements);
    free(F.elements);

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}