/*
Overall Performance:
    RUNTIME:    2.72 ms
    GFLOPS:     1121.99
    STATUS:     Accepted
    SUBMITTED:  12/11/2025, 7:31:01 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.79         | 1037.44  |
| N=32768, K=8191      | 0.54         | 997.88   |
| N=131072, K=8191     | 1.84         | 1170.08  |
| N=524288, K=8191     | 6.70         | 1282.57  |

Description: 1D convolution using cooperative loading into shared memory with kernel in readonly cache.
Readme: This implementation optimizes the 772_71 GFLOP Solution by introducing a readonly cache for the kernel.
        By storing the kernel in the constant memory, we can further reduce global memory accesses during the 
        convolution, leading to further performance improvements and achieving over 1 TFLOP on the T4 GPU.
        The limitation here is that we can now only handle kernels up to 8192 elements due to constant memory size limits.
*/

#include <cuda_runtime.h>
#define BASE_THREAD_NUM 1024
#define MAX_K 8192
__constant__ float cKernel[MAX_K];

__global__ void ConvKernel(const float* A, const float* B, float* out, const size_t out_size, const size_t kernel_size){
    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread needs to cooperatively load A into this
    // this will be sized to the appropriate block halo region
    // by the launcher
    const int radius = (kernel_size - 1) / 2;
    const int region_size = blockDim.x + (kernel_size - 1);
    extern __shared__ float block_shared_A[];

    // shared[0] needs to correspond to the blockbase - radius
    const int baseofs = blockIdx.x * blockDim.x - radius;
    
    // Each thread loads multiple elements of the shared tile
    // each thread should start with its own tidx, and load from the
    // base offset + tidx, and then increment by threadblock size until
    // we cover the whole region
    for (int s = tidx; s < region_size; s += blockDim.x) {
        int gidx = baseofs + s;
        float v = 0.0f;
        if (gidx >= 0 && gidx < out_size) {
            v = A[gidx];
        }
        block_shared_A[s] = v;
    }
    __syncthreads();

    if(gid >= out_size){
        return;
    }

    // each thread's window starts at its tidx in shared (shared[i] = A[i - radius])
    float sum = 0.0f;
    int base = tidx;
    for (int j = 0; j < kernel_size; ++j) {
        sum += block_shared_A[base + j] * cKernel[j];
    }
    out[gid] = sum;
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(cKernel, B, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    const size_t shm_bytes = (BASE_THREAD_NUM + (K - 1)) * sizeof(float);
    ConvKernel<<<grid, block, shm_bytes>>>(A, B, C, (const size_t) N, (const size_t) K);
}
