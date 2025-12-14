/*
Overall Performance:
    RUNTIME:    4.05 ms
    GFLOPS:     782.03
    STATUS:     Accepted
    SUBMITTED:  12/13/2025, 1:22:39 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution
    TEST CASES: 4/4 passed

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.48         | 775.64   |
| N=32768, K=8191      | 0.68         | 791.27   |
| N=131072, K=8191     | 2.66         | 806.99   |
| N=524288, K=8191     | 11.39        | 754.24   |

Submitted Code Defines:
    #define BASE_THREAD_NUM 1024

README: Got rid of constant memory for kernel, as it felt like a hack / bench-specific optimization.
        Instead, we load the kernel cooperatively into shared memory in tiles of BASE_THREAD_NUM size.
        This allows us to handle arbitrarily large kernels without relying on constant memory.
        The rest of the logic is similar to the previous coop load signal version, with threads
        cooperatively loading the input signal into shared memory.
*/


#include <cuda_runtime.h>
#define BASE_THREAD_NUM 1024

__global__ void ConvKernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ out, 
    const size_t out_size, 
    const size_t kernel_size
){
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
    int s;
    for (s = tidx; s < region_size; s += blockDim.x) {
        int gidx = baseofs + s; //leftmost element needed for this thread
        float v = 0.0f;
        if (gidx >= 0 && gidx < out_size) {
            v = A[gidx];
        }
        block_shared_A[s] = v;
    }
    __syncthreads();

    float sum = 0.0f;
    __shared__ float shB[BASE_THREAD_NUM]; 
    for(s=0; s < kernel_size; s += BASE_THREAD_NUM){
        __syncthreads();
        // fetch the kernel tile
        if((s+tidx) < kernel_size){
            shB[tidx] = B[s+tidx];
        } else {
            shB[tidx] = 0;
        }
        __syncthreads();
    
        // each thread's window starts at its tidx in shared (shared[i] = A[i - radius])
        for (int j = s; j < s+BASE_THREAD_NUM; ++j) {
            // j-s is maximum BASE_THREAD_NUM-1, so shB access is safe
            // we have to make sure that j doesn't go out of bounds on 
            // A though. a size is out_size + 2*radius == osize+ksize
            if(tidx+j < region_size){
                sum += block_shared_A[tidx+j] * shB[j-s];
            }
        }

    }

    if(gid < out_size){
        out[gid] = sum;
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    const size_t shm_bytes = (BASE_THREAD_NUM + (K - 1)) * sizeof(float);
    ConvKernel<<<grid, block, shm_bytes>>>(A, B, C, (const size_t) N, (const size_t) K);
}
