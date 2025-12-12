/*
Overall Performance:
    RUNTIME:    7.98 ms
    GFLOPS:     454.42
    STATUS:     Accepted
    SUBMITTED:  12/11/2025, 3:20:47 PM

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 2.36         | 467.47   |
| N=32768, K=8191      | 1.04         | 516.49   |
| N=131072, K=8191     | 4.52         | 475.19   |
| N=524288, K=8191     | 24.01        | 358.53   |

Description: 1D convolution using readonly cache for kernel.
Readme: This implementation uses the __constant__ memory space to store the kernel (B array).
        Since the kernel is read-only and accessed by all threads, using constant memory allows
        for more efficient / explicit caching and broadcasted reads, improving performance over global memory accesses.
        Only does a little bit better than the naive implementation though (~10 GFLOPS).
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
    out[gid] = 0.0;
    size_t ofs = (kernel_size-1)/2;
    for(size_t j=0; j<kernel_size; ++j){
        size_t aidx = gid + j - ofs;
        float val_a = 0.0;
        if(aidx>=0 && aidx<out_size){
            val_a = A[aidx];
        }
        out[gid] += val_a * cKernel[j];
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(cKernel, B, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    ConvKernel<<<grid, block>>>(A, C, (const size_t) N, (const size_t) K);
}
