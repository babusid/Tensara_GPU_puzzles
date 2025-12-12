/*
Overall Performance:
    RUNTIME:    5.48 ms
    GFLOPS:     616.25
    STATUS:     Accepted
    SUBMITTED:  12/11/2025, 6:33:53 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.82         | 627.25   |
| N=32768, K=8191      | 0.85         | 628.44   |
| N=131072, K=8191     | 3.21         | 668.65   |
| N=524288, K=8191     | 16.02        | 540.64   |

Description: 1D convolution using naive approach with kernel stored in global memory and optimized summation loop.
Readme: This implementation performs 1D convolution by directly accessing the kernel from global memory.
        This benchmarks faster than the one with readonly memory, maybe due to the lack of the cudaMemcpyToSymbol overhead.
*/

#include <cuda_runtime.h>
#define BASE_THREAD_NUM 128
#define MAX_K 8192
__constant__ float cKernel[MAX_K];

__global__ void ConvKernel(const float* A, const float* B, float* out, const size_t out_size, const size_t kernel_size){
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
        sum += val_a * B[j];
    }
    out[gid] = sum;
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    // cudaMemcpyToSymbol(cKernel, B, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    ConvKernel<<<grid, block>>>(A, B, C, (const size_t) N, (const size_t) K);
}
