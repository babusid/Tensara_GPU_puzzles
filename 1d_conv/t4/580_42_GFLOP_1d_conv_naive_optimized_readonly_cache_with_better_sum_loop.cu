/*
Overall Performance:
    RUNTIME:    5.41 ms
    GFLOPS:     580.42
    STATUS:     Accepted
    SUBMITTED:  12/11/2025, 6:33:06 PM

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 2.10         | 533.81   |
| N=32768, K=8191      | 0.95         | 563.60   |
| N=131072, K=8191     | 3.25         | 660.26   |
| N=524288, K=8191     | 15.32        | 563.99   |

Description: 1D convolution using readonly cache for kernel with improved summation loop.
Readme: This implementation builds upon the previous readonly cache version by optimizing the summation loop,
        storing the intermediate sum in a local variable before writing it to global memory. 
        By reducing the number of global memory writes, we achieve better performance and higher GFLOPS 
        compared to the earlier version.
*/
#include <cuda_runtime.h>
#define BASE_THREAD_NUM 128
#define MAX_K 8192
__constant__ float cKernel[MAX_K];

__global__ void ConvKernel(const float* A, float* out, const size_t out_size, const size_t kernel_size){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= out_size){
        return;
    }
    size_t ofs = (kernel_size-1)/2;
    float sum = 0;
    for(size_t j=0; j<kernel_size; ++j){
        int aidx = gid + j - ofs;
        float val_a = 0.0;
        if(aidx>=0 && aidx<out_size){
            val_a = A[aidx];
        }
        sum += val_a * cKernel[j];
    }
    out[gid] = sum;
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(cKernel, B, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    ConvKernel<<<grid, block>>>(A, C, (const size_t) N, (const size_t) K);
}
