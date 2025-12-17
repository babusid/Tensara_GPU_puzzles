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
                             const int K) {

  // These give the base output indices for this thread
  // Each thread computes ELEMENTS_PER_THREAD x ELEMENTS_PER_THREAD elements
  // Each block computes a TILESIZE x TILESIZE tile of the output matrix
  int out_col = blockIdx.x * TILESIZE + (threadIdx.x * ELEMENTS_PER_THREAD);
  int out_row = blockIdx.y * TILESIZE + (threadIdx.y * ELEMENTS_PER_THREAD);

  float partial_sums[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD] = {0};

  // Shared memory for tiles of A and B
  __shared__ float shared_a[TILESIZE][TILESIZE];
  __shared__ float shared_b[TILESIZE][TILESIZE];

  for (int tile_k_start = 0; tile_k_start < K; tile_k_start += TILESIZE) {

    // Shared memory indices for loading input tiles into shared memory
    // we don't have to worry about block offsetting here, every block has its
    // own shared memory, and should be symmetric.
    int shared_A_input_tile_row = threadIdx.y * ELEMENTS_PER_THREAD;
    int shared_B_input_tile_col = threadIdx.x * ELEMENTS_PER_THREAD;

    // Global indices for loading input tiles
    // We use block index to find the patch in the output matrix this
    // threadblock is responsible for and then offset by thread index to find
    // which small patch of that ouptut tile this thread is responsible for
    // computing
    int global_A_input_tile_row =
        (blockIdx.y * TILESIZE) +
        (shared_A_input_tile_row); // this is the base row for A tile load
    int global_B_input_tile_col =
        (blockIdx.x * TILESIZE) +
        (shared_B_input_tile_col); // this is the base col for B tile load

    // we need to get a 2x2 patch of values from A, where
    // global_A_input_tile_row corresponds to the top left of the patch. This
    // patch should be stored in shared_a at threadIdx.y*ELEMENTS_PER_THREAD,
    // threadIdx.x*ELEMENTS_PER_THREAD
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
        // double loop allows us to do a 2x2 patch

        // take the base global row for A tile load, and offset by i to get the
        // correct row in the patch to load
        int a_row = global_A_input_tile_row + i;

        // tile_k_start gives the starting column for the A input tile load.
        // Offset by threadIdx.x*ELEMENTS_PER_THREAD to get the base col for
        // this thread then offset by j to get the correct col in the patch to
        // load
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
        int b_col = global_B_input_tile_col + j;
        if (b_row < K && b_col < N) {
          shared_b[threadIdx.y * ELEMENTS_PER_THREAD + i]
                  [shared_B_input_tile_col + j] = b[b_row * N + b_col];
        } else {
          shared_b[threadIdx.y * ELEMENTS_PER_THREAD + i]
                  [shared_B_input_tile_col + j] = 0.0f;
        }
      }
    }
    __syncthreads(); // ensure the tile is loaded before computation

    // Compute partial sums for the TILESIZE x TILESIZE tile.
    // To do this, we want to prefetch 2x2 patches from the shared memory input
    // tiles

    for (int k = 0; k < TILESIZE; k += ELEMENTS_PER_THREAD) {
      // Prefetch a 2x2 patch from shared_a and shared_b
      float a_values[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
      float b_values[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
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