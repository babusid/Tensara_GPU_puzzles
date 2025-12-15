/*
SUBMISSION RESULT: ACCEPTED 
Avg runtime: 720.97 ms
Avg gflops: 716.30

Detailed Results:
Test Case                          GFLOPS    Runtime (ms)          Status
───────────────────────────────────────────────────────────────────────────
✓ 1   4096x4096 x 4096x4096        722.52        190.2288          PASSED
✓ 2   8192x8192 x 8192x4096        723.79        759.5620          PASSED
✓ 3   4096x4096 x 4096x8192        707.49        388.5482          PASSED
✓ 4   8192x8192 x 8192x8192        711.42       1545.5266          PASSED

README: This implementation uses shared memory to optimize matrix multiplication.
        Each thread block computes a TILESIZE x TILESIZE sub-matrix of the output.
        Threads within a block cooperatively load tiles of the input matrices
        into shared memory, reducing global memory accesses and improving performance.
*/
#include <cuda_runtime.h>
#define TILESIZE 32

__global__ void CoopFetchMatmulKernel(
  const float* __restrict__ a, 
  const float* __restrict__ b,
  float* __restrict__ out, 
  const int M,
  const int N, 
  const int K
) {
  int out_j = blockIdx.x * TILESIZE + threadIdx.x; // Column index (0..N-1)
  int out_i = blockIdx.y * TILESIZE + threadIdx.y; // Row index (0..M-1)
  
  const bool valid_i = (out_i < M);
  const bool valid_j = (out_j < N);
  float partial_sum = 0.0f; 

  // Shared memory for tiles of A and B
  __shared__ float shared_a[TILESIZE][TILESIZE];
  __shared__ float shared_b[TILESIZE][TILESIZE];

  for (size_t tile_k_start = 0; tile_k_start < K; tile_k_start += TILESIZE) { 
    
    
    // load A and B patches into shmem
    size_t global_A_k_index = tile_k_start + threadIdx.x; 
    if (valid_i && global_A_k_index < K) {
      shared_a[threadIdx.y][threadIdx.x] = a[out_i * K + global_A_k_index];
    } else {
      shared_a[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    size_t global_B_k_index = tile_k_start + threadIdx.y; 
    if (global_B_k_index < K && valid_j) {
      shared_b[threadIdx.y][threadIdx.x] = b[global_B_k_index * N + out_j];
    } else {
      shared_b[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads(); 

    // partial sum
    for (int it = 0; it < TILESIZE; ++it) {
      float a_element = shared_a[threadIdx.y][it];
      float b_element = shared_b[it][threadIdx.x];
      partial_sum += a_element * b_element;
    }
    __syncthreads();
  }
  
  if (valid_i && valid_j) {
    out[out_i * N + out_j] += partial_sum;
  }
}


// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
  // ceiling div of columns by tile size. X because we iterate across the columns of B
  int grid_x = (n + TILESIZE - 1) / TILESIZE;   
  
  // ceiling div of M by tile size. Y because we iterate down the rows of A
  int grid_y = (m + TILESIZE - 1) / TILESIZE;  
  
  dim3 gridSize(grid_x, grid_y);  // number of thread blocks
  dim3 blockSize(TILESIZE, TILESIZE);
  CoopFetchMatmulKernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, (int)m, (int)n, (int)k);
}