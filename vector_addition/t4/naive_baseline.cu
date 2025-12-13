/*
Overall Performance:
    RUNTIME:    11.54 ms
    GFLOPS:     20.57
    STATUS:     Accepted
    SUBMITTED:  12/12/2025, 12:21:27 PM
    DEVICE:     T4
    PROBLEM:    Vector Addition
    TEST CASES: 7/7 passed

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| n = 2^20             | 0.59         | 16.70    |
| n = 2^22             | 0.20         | 20.49    |
| n = 2^23             | 0.40         | 20.92    |
| n = 2^25             | 1.58         | 21.19    |
| n = 2^26             | 3.11         | 21.55    |
| n = 2^29             | 24.65        | 21.78    |
| n = 2^30             | 50.21        | 21.38    |

Submitted Code Defines:
    #define BASE_THREAD_NUM 128
*/

#include <cuda_runtime.h>
#define BASE_THREAD_NUM 128

__global__ void AddKernel(const float* d_input1, const float* d_input2, float* d_output, const size_t n){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n){
        return;
    }
    d_output[gid] = d_input1[gid] + d_input2[gid];
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    size_t num_blocks = (n + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    AddKernel<<<grid, block>>>(d_input1, d_input2, d_output, (const size_t) n);
}