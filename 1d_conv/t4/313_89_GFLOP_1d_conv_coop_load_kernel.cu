/*
Overall Performance:
    RUNTIME:    9.96 ms
    GFLOPS:     313.89

Benchmark Results:
| Test Case            | Runtime (ms) | GFLOPS   |
| -------------------- | ------------ | -------- |
| N=65536, K=8191      | 3.22         | 339.61   |
| N=32768, K=8191      | 1.87         | 287.06   |
| N=131072, K=8191     | 6.82         | 320.01   |
| N=524288, K=8191     | 27.94        | 308.87   |


Description: 1D convolution using cooperative loading of kernel into shared memory.
Readme: This one uses cooperative loading of the kernel into shared memory to reduce global memory accesses.
        However, it actually does worse than the naive kernel because of the additional synchronization overhead,
        and the fact that in the naive kernel, every thread accesses the **same** kernel values in the **same order**,
        which allows the GPU to effectively cache these values in L1/L2 cache and causes broadcasted loads, 
        making the memory accesses for the B array very efficient already.
        Will likely be more beneficial to cooperatively load the A array, considering that each thread accesses different
        values from A. Will also have to be careful about this in order to do better, considering that each thread is accessing 
        adjacent values from A, so we are likely to already have coalesced reads.
*/


#include <cuda_runtime.h>
#define BASE_THREAD_NUM 128

__global__ void ConvKernel(const float* A, const float* B, float* out, const size_t out_size, const size_t kernel_size){
    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;
    size_t gid = bidx * blockDim.x + tidx;
    if(gid < out_size){
        out[gid] = 0.0;
    }

    // ceiling div of kernel size into threadblock
    // sized chunks
    // we want to coop load chunk at a time and use them for the conv
    size_t kernel_chunks = (kernel_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; 
    __shared__ float shB[BASE_THREAD_NUM]; 
    
    for(size_t chunk=0; chunk<kernel_chunks; ++chunk){
        // optimization 1: cooperatively fetch the kernel (B) into shared memory
        size_t load_idx = chunk*BASE_THREAD_NUM + tidx;
        __syncthreads();
        if(load_idx < kernel_size){
            shB[tidx] = B[load_idx];
        }
        __syncthreads();
        
        // do the convolution operation
        size_t A_ofs = (kernel_size-1)/2;
        for(
            size_t j=(chunk*BASE_THREAD_NUM); 
            j<min(kernel_size, (chunk + 1) * BASE_THREAD_NUM); 
            ++j
        ){
            int aidx = (gid + j - A_ofs);
            float val_a = 0.0;
            if(aidx>=0 && aidx<(long)out_size){ 
                val_a = A[gid + j - A_ofs];
            }
            if(gid < out_size){
                out[gid] += val_a * shB[j - (chunk*BASE_THREAD_NUM)];
            }
        }
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    size_t num_blocks = (N + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // how many blocks to launch
    dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
    dim3 grid = dim3(num_blocks, 1, 1);
    ConvKernel<<<grid, block>>>(A, B, C, (const size_t) N, (const size_t) K);
}
