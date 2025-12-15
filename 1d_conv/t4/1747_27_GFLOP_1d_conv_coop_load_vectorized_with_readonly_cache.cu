/*
Overall Performance:
    RUNTIME:    1.71 ms
    GFLOPS:     1747.27
    STATUS:     Accepted
    SUBMITTED:  12/13/2025, 8:07:33 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution
    TEST CASES: 4/4 passed

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 0.71         | 1677.31  |
| N=32768, K=8191      | 0.32         | 1677.21  |
| N=131072, K=8191     | 1.22         | 1760.65  |
| N=524288, K=8191     | 4.58         | 1873.90  |

Submitted Code Defines:
    #define BASE_THREAD_NUM 448
    #define MAX_K 8192

README: This implementation optimizes the cooperative loading of the input signal into shared memory
        by utilizing vectorized loads with float4. It also uses a vectorized store with float4. 
        This should decrease the number of both load and store instructions, leading to improved overall throughput. 
        The kernel still uses constant memory forthe convolution kernel for fast access during the convolution computation, 
        I can't yet find a way to make that faster / better. 
        This kernel (at the time of writing), is at #6 on the global leaderboard for 1D convolution on T4.
*/
#include <cuda_runtime.h>
#define BASE_THREAD_NUM 448
#define MAX_K 8192
__constant__ float cKernel[MAX_K];

__global__ void ConvKernel(const float* __restrict__ A, float* __restrict__ out, 
                           const size_t out_size, const size_t kernel_size){
    const int tidx = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int radius = (kernel_size - 1) / 2;
    const int region_size = blockDim.x + (kernel_size - 1);
    
    extern __shared__ float block_shared_A[];
    
    //shared[0] should correspond to the start of the halo region, 
    // ie block's first thread position in the global array - half the kernel size
    const int baseofs = (blockIdx.x * blockDim.x) - radius;
    
        // Vectorized cooperative load using float4
    const int region_size_aligned = (region_size / 4) * 4;
    
    // Load using float4 for better memory throughput
    for (int s = tidx * 4; s < region_size_aligned; s += blockDim.x * 4) {
        int gidx = baseofs + s;
        
        // init float4 with all zeros
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 
        
        // Check bounds for all 4 elements, out_size is same as A size.
        bool valid0 = (gidx >= 0) && (gidx < out_size);
        bool valid1 = (gidx + 1 >= 0) && (gidx + 1 < out_size);
        bool valid2 = (gidx + 2 >= 0) && (gidx + 2 < out_size);
        bool valid3 = (gidx + 3 >= 0) && (gidx + 3 < out_size);
        
        if (valid0) v4.x = __ldg(&A[gidx]);
        if (valid1) v4.y = __ldg(&A[gidx + 1]);
        if (valid2) v4.z = __ldg(&A[gidx + 2]);
        if (valid3) v4.w = __ldg(&A[gidx + 3]);
        
        reinterpret_cast<float4*>(&block_shared_A[s])[0] = v4;
    }
    
    // Handle leftover elements that werent in the aligned region for float4 loading
    for (int s = region_size_aligned + tidx; s < region_size; s += blockDim.x) {
        int gidx = baseofs + s;
        float v = 0.0f;
        if (gidx >= 0 && gidx < out_size) {
            v = __ldg(&A[gidx]);
        }
        block_shared_A[s] = v;
    }

    __syncthreads();
    
    if(gid >= out_size) return;
    
    // conv loop
    float sum = 0.0f;
    const float* local_A = &block_shared_A[tidx];
    int j = 0;
    #pragma unroll 8
    for (; j < kernel_size; ++j) {
        sum += local_A[j] * cKernel[j];
    }
    
    out[gid] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(cKernel, B, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    dim3 block(BASE_THREAD_NUM);
    dim3 grid(num_blocks);
    const size_t shm_bytes = (BASE_THREAD_NUM + (K - 1)) * sizeof(float);
    ConvKernel<<<grid, block, shm_bytes>>>(A, C, N, K);
}