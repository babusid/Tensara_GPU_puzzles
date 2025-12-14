/*
Overall Performance:
    RUNTIME:    3.38 ms
    GFLOPS:     838.61
    STATUS:     Accepted
    SUBMITTED:  12/13/2025, 1:27:43 PM
    DEVICE:     T4
    PROBLEM:    1D Convolution
    TEST CASES: 4/4 passed

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 1.43         | 776.52   |
| N=32768, K=8191      | 0.68         | 789.40   |
| N=131072, K=8191     | 2.66         | 808.25   |
| N=524288, K=8191     | 8.76         | 980.27   |

Submitted Code Defines:
    #define BASE_THREAD_NUM 1024

README: Builds off 782_03_GFLOP_1d_conv_coop_load_both.cu. Optimizes by using only int arithmetic instead of size_t where possible.
        Also, earlier kernel relies on convkernel tile buffer being zeroed beyond the boundaries of the kernel itself (which happens
        on the last tile potentially). We also keep a boundary check. Only time when the boundary check is false is when j >= kernel_size.
        Instead, we just pre-clamp the end of j loop to kernel_size, and avoid the boundary check inside the loop entirely.
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
    const int tidx = (int)  threadIdx.x;
    const int bidx = (int)  blockIdx.x;
    const int gid  = (int)  blockIdx.x * blockDim.x + threadIdx.x;
    const int N    = (int)  out_size;
    const int K    = (int)  kernel_size;
    // each thread needs to cooperatively load A into this
    // this will be sized to the appropriate block halo region
    // by the launcher
    const int radius = (K - 1) / 2;
    // width of threadblock plus radius on each side
    const int region_size = blockDim.x + (K - 1); 
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
        if (gidx >= 0 && gidx < N) {
            v = A[gidx];
        }
        block_shared_A[s] = v;
    }
    __syncthreads();

    float sum = 0.0f;
    __shared__ float shB[BASE_THREAD_NUM]; 
    for(s=0; s < K; s += BASE_THREAD_NUM){
        __syncthreads();
        // fetch the kernel tile
        if((s+tidx) < K){
            shB[tidx] = B[s+tidx];
        } 
        __syncthreads();
    
        // each thread's window starts at its tidx in shared (shared[i] = A[i - radius])
        int j_end = min((int)kernel_size, s + BASE_THREAD_NUM);
        for (int j = s; j < j_end; ++j) {
            // j-s is maximum BASE_THREAD_NUM-1, so shB access is safe
            // we have to make sure that j doesn't go out of bounds on 
            // A though. a size is out_size + 2*radius == osize+ksize
            sum += block_shared_A[tidx + j] * shB[j - s];
        }

    }

    if(gid < N){
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
