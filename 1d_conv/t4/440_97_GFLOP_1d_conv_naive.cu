/*
Benchmark Results
| Test Case | Runtime (ms) | GFLOPS |
| --- | --- | --- |
| N=65536, K=8191 | 4.24 | 433.61 |
| N=32768, K=8191 | 1.20 | 446.21 |
| N=131072, K=8191 | 4.40 | 488.04 |
| N=524288, K=8191 | 21.81 | 396.03 |

Average: 440.97GFLOPS

Description: Basic 1d conv with left/right padding.
*/

#include <cuda_runtime.h>
#define BASE_THREAD_NUM 128

__global__ void ConvKernel(const float* A, const float* B, float* out, const size_t out_size, const size_t kernel_size){
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
        out[gid] += val_a * B[j];
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    ConvKernel<<<grid, block>>>(A, B, C, (const size_t) N, (const size_t) K);
}
