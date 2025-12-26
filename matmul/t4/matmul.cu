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

README: This implementation uses shared memory to optimize matrix
multiplication. Each thread block computes a TILESIZE x TILESIZE sub-matrix of
the output. Threads within a block cooperatively load tiles of the input
matrices into shared memory, reducing global memory accesses and improving
performance.
*/
#include <cuda_runtime.h>
// #define TILESIZE 32
#define BLOCKSIZE                                                              \
  8 // number of threads in each dimension of a block (block will have 8x8 = 64
    // threads)
#define ELEMENTS_PER_THREAD                                                    \
  2 // how many output elements each thread computes (2x2)
#define TILESIZE                                                               \
  (BLOCKSIZE * ELEMENTS_PER_THREAD) // how many output elements calculated by a
                                    // threadblock in a dimension

__global__ void MatmulKernel(const float *__restrict__ a,
                             const float *__restrict__ b,
                             float *__restrict__ out, const int M, const int N,
                             const int K) 
{

  float partial_sums[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD] = {0};

  // Shared memory for tiles of A and B
  __shared__ float shared_a[TILESIZE][TILESIZE];
  __shared__ float shared_b[TILESIZE][TILESIZE];
  
  // These give the base output indices for this thread
  // Each thread computes ELEMENTS_PER_THREAD x ELEMENTS_PER_THREAD elements
  // Each block computes a TILESIZE x TILESIZE tile of the output matrix
  int out_row = blockIdx.y * TILESIZE + (threadIdx.y * ELEMENTS_PER_THREAD);
  int out_col = blockIdx.x * TILESIZE + (threadIdx.x * ELEMENTS_PER_THREAD);

  // Shared memory indices to indicate where to store the loaded global data.
  // Acts as base offsets for this thread to store into the sub-patch of shared mem
  // it is responsible for.
  int shared_A_input_tile_row = threadIdx.y * ELEMENTS_PER_THREAD;
  int shared_B_input_tile_col = threadIdx.x * ELEMENTS_PER_THREAD;
    
  #pragma unroll
  for (int tile_k_start = 0; tile_k_start < K; tile_k_start += TILESIZE) {
  // we need to get a TILESIZExTILESIZE input tile of values from A,B
  // each thread needs to load a sub-patch of this input tile
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      #pragma unroll
      for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {

        // take the base global row for A tile load, and offset by i to get the
        // correct row in the patch to load from. The output's row is the same as the row in matrix A.
        int a_row = out_row + i;

        // tile_k_start gives the starting column for the full input tile from
        // A, since we are iterating left to right across A. We offset this with
        // the thread offset to isolate the starting column for the sub-patch
        // this thread is responsible for. We then offset with to iterate from
        // that starting column forward as we progress through the loop
        int a_col = tile_k_start + (threadIdx.x * ELEMENTS_PER_THREAD) + j;
        if (a_row < M && a_col < K) {
          shared_a[shared_A_input_tile_row + i]
                  [threadIdx.x * ELEMENTS_PER_THREAD + j] =
                      a[a_row * K + a_col];
        } else {
          shared_a[shared_A_input_tile_row + i]
                  [threadIdx.x * ELEMENTS_PER_THREAD + j] = 0.0f;
        }

        // similar logic for B matrix
        int b_row = tile_k_start + (threadIdx.y * ELEMENTS_PER_THREAD) + i;
        int b_col = out_col + j;
        if (b_row < K && b_col < N) {
          shared_b[threadIdx.y * ELEMENTS_PER_THREAD + i]
                  [shared_B_input_tile_col + j] = b[b_row * N + b_col];
        } else {
          shared_b[threadIdx.y * ELEMENTS_PER_THREAD + i]
                  [shared_B_input_tile_col + j] = 0.0f;
        }
      }
    }  
  }
  __syncthreads(); // ensure that the input tiles are fully loaded

  // This thread is responsible for a patch of the output tile of dim
  // ELEMENTS_PER_THREAD x ELEMENTS_PER_THREAD. 
  // horizontally adjacent psums use the same row, 
  // vertically adjacent psums use the same column, maybe some reuse opportunity here later.
  for(int i = 0; i<ELEMENTS_PER_THREAD;++i){
    for(int j = 0; j<ELEMENTS_PER_THREAD;++j){
      int a_tile_row = shared_A_input_tile_row + i;
      int b_tile_col = shared_B_input_tile_col + j;
      for(int k = 0; k<TILESIZE; ++k){
        partial_sums[i][j] += shared_a[a_tile_row][k] * shared_b[k][b_tile_col];
      }
    }
  }

  // write to global
  for(int i = 0; i<ELEMENTS_PER_THREAD;++i){
    for(int j = 0; j<ELEMENTS_PER_THREAD;++j){
      if(out_row + i < M && out_col + j < N){
        out[(out_row+i)*N + (out_col+j)] = partial_sums[i][j];
      }
    }
  }

}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float *input_a, const float *input_b,
                         float *output_c, size_t m, size_t n, size_t k) {
  // ceiling div of columns by output tile size. X because we iterate across the
  // columns of B output tile size is used to determine number of blocks needed,
  // because each block computes a tile of output
  int grid_x = (n + TILESIZE - 1) / TILESIZE;

  // ceiling div of M by tile size. Y because we iterate down the rows of A
  int grid_y = (m + TILESIZE - 1) / TILESIZE;

  dim3 gridSize(grid_x, grid_y); // number of thread blocks
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
  MatmulKernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, (int)m,
                                        (int)n, (int)k);
}

