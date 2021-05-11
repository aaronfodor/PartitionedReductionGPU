#include "cuda_functions.h"

// number of threads in a block
const int BLOCK_SIZE = 32;

__global__ void partitioned_reduce_kernel(float* device_input){
    int idx = threadIdx.x;
    int currentIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(idx < s){
            device_input[currentIdx] = max(device_input[currentIdx], device_input[currentIdx + s]);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        device_input[blockIdx.x] = device_input[currentIdx];
    }
}

float CUDA_FUNCTIONS::partitioned_reduction(float* input_array, const size_t dataSize){
    float* device_input = nullptr;
    float result;

    int numBlocks = ceil(dataSize / BLOCK_SIZE) + 1;

    dim3 dimGrid = numBlocks;
    dim3 dimBlock = BLOCK_SIZE;
    int numMergeIterations = ceil((log(numBlocks) / log(BLOCK_SIZE)) + 1);

    cudaMalloc(&device_input, sizeof(int) * dataSize);
    cudaMemcpy(device_input, input_array, sizeof(int) * dataSize, cudaMemcpyHostToDevice);

    for(unsigned int i = 0; i < numMergeIterations; i++){
        partitioned_reduce_kernel<<<dimGrid, dimBlock>>>(device_input);
    }

    cudaMemcpy(&result, device_input, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_input);

    return result;
}
