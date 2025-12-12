/*
Overall Performance:
    RUNTIME:    3.71 ms
    GFLOPS:     772.71
    STATUS:     Accepted
    SUBMITTED:  12/11/2025, 7:15:26 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.52         | 725.29   |
| N=32768, K=8191      | 0.73         | 732.72   |
| N=131072, K=8191     | 2.86         | 750.43   |
| N=524288, K=8191     | 9.74         | 882.42   |

Description: 1D convolution using cooperative loading into shared memory for input signal.
Readme: This implementation optimizes the 1D convolution by having threads within a block cooperatively
        load the necessary segment of the input signal into shared memory. This reduces global memory
        accesses during the convolution operation, leading to improved performance and higher GFLOPS.
        Note the very high threadblock size of 1024. This is because of empirical testing which showed that
        due to very large kernel sizes, the dynamic shared memory sizes were also very large. Even with a small 
        threadblock size (ie. 128, 256), the shared memory usage per SM was still so high that only 1 block could be
        scheduled per SM. By increasing the threadblock size to 1024, we can use all 32 warps per SM and achieve higher 
        overall throughput.

        blocksize 128, ksize= 8192 -> shmem = 8192*4 + 128*4 ~ 33 KB -> 1 block per SM, 4 warps active
        blocksize 1024, ksize= 8192 -> shmem = 8192*4 + 1024*4 ~ 36 KB -> 1 blocks per SM, 32 warps active
*/


#include <cuda_runtime.h>
#define BASE_THREAD_NUM 1024

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
        sum += block_shared_A[base + j] * B[j];
    }
    out[gid] = sum;
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    const size_t shm_bytes = (BASE_THREAD_NUM + (K - 1)) * sizeof(float);
    ConvKernel<<<grid, block, shm_bytes>>>(A, B, C, (const size_t) N, (const size_t) K);
}
