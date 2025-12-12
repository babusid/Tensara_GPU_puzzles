/*
Overall Performance:
    RUNTIME:    2.52 ms
    GFLOPS:     1138.24
    STATUS:     Accepted
    SUBMITTED:  12/12/2025, 11:58:36 AM
    DEVICE:     T4
    PROBLEM:    1D Convolution

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.32         | 1031.81  |
| N=32768, K=8191      | 0.54         | 1001.83  |
| N=131072, K=8191     | 1.83         | 1173.57  |
| N=524288, K=8191     | 6.38         | 1345.75  |

Description: 1D convolution using cooperative loading into shared memory with kernel in readonly cache.
Readme: This implementation builds upon the 1121_99 GFLOP Solution by further optimizing the access pattern to 
        the block shared memory / global memory, by pre-calculating the base pointer into shared memory for each thread.
        This leads to less redundant arithmetic instructions and slightly better performance, achieving 1138 GFLOPS on the T4 GPU.
*/
#include <cuda_runtime.h>
#define BASE_THREAD_NUM 1024
#define MAX_K 8192 // max kernel support is 8k
__constant__ float cKernel[MAX_K];

__global__ void ConvKernel(const float* A, const float* B, float* out, const size_t out_size, const size_t kernel_size){
    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread needs to cooperatively load A into this
    // this will be sized to the appropriate block halo region
    // by the launcher
    const int radius = (kernel_size - 1) / 2;
    // width of threadblock plus radius on each side
    const int region_size = blockDim.x + (kernel_size - 1); 
    extern __shared__ float block_shared_A[];

    // shared[0] needs to correspond to the blockbase - radius
    const int baseofs = (blockIdx.x * blockDim.x) - radius;
    
    // Each thread loads multiple elements of the shared tile
    // each thread should start with its own tidx, and load from the
    // base offset + tidx, and then increment by threadblock size until
    // we cover the whole region
    for (int s = tidx; s < region_size; s += blockDim.x) {
        int gidx = baseofs + s; //leftmost element needed for this thread
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
    float* local_A = &block_shared_A[base];
    for (int j = 0; j < kernel_size; ++j) {
        sum += local_A[j] * cKernel[j];
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
