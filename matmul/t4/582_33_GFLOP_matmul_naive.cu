/**

BENCHMARK RESULTS 
═════════════════════════════════════════════════════════════════
Metric                              Value
─────────────────────────────────────────────────────────────────
Total Benchmarks:                       4
Average GFLOPS:                    582.33
Average Runtime:                   888.82 ms
═════════════════════════════════════════════════════════════════

Detailed Results:
Test Case                          GFLOPS    Runtime (ms)          Status
───────────────────────────────────────────────────────────────────────────
✓ 1   4096x4096 x 4096x4096        587.08        234.1315          PASSED
✓ 2   8192x8192 x 8192x4096        584.34        940.8328          PASSED
✓ 3   4096x4096 x 4096x8192        581.52        472.7245          PASSED
✓ 4   8192x8192 x 8192x8192        576.39       1907.5777          PASSED

README: This is a naive implementation of matrix multiplication in CUDA.
        Each thread computes one element of the output matrix by performing
        a dot product of the corresponding row from matrix A and column from matrix B.
        This implementation does not use shared memory or any advanced optimization techniques,
        making it straightforward but memory-bound. It serves as a baseline for performance comparison.
*/

#include <cuda_runtime.h>
#define SQRTBLOCKSIZE 32

__global__ void MatmulKernel(
  const float* __restrict__ a, 
  const float* __restrict__ b,
  float* __restrict__ out, 
  const int M,
  const int N, 
  const int K
) {
  /**
   * Naive CUDA Kernel for matrix multiply.
   * Each thread corresponds to an output element, 
   * and runs a N-length loop with no shared memory. 
   * Heavily memory-bound, but simple.
   */
  size_t out_j = blockIdx.x * blockDim.x + threadIdx.x; // columns correspond to x index
  size_t out_i = blockIdx.y * blockDim.y + threadIdx.y; // rows correspond to y index
  
  // bounds check out_i, out_j
  if (out_i >= M) {
    return;
  }
  if (out_j >= N) {
    return;
  }
  float sum = 0.0f;
  for (size_t it = 0; it < K; ++it) { // Loop over 'K' dimension
    sum += a[out_i * K + it] * b[it * N + out_j];
  }
  out[out_i * N + out_j] = sum;
}


// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
  // ceiling div of columns by tile size. X because we iterate across the columns of B
  int grid_x = (n + SQRTBLOCKSIZE - 1) / SQRTBLOCKSIZE;   
   // ceiling div of M by tile size. Y because we iterate down the rows of A
  int grid_y = (m + SQRTBLOCKSIZE - 1) / SQRTBLOCKSIZE;  
  
  dim3 gridSize(grid_x, grid_y);  // number of thread blocks
  dim3 blockSize(SQRTBLOCKSIZE, SQRTBLOCKSIZE);
  MatmulKernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, (int)m, (int)n, (int)k);
}