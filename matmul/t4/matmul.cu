#include <cuda_runtime.h>
#define BLOCKSIZE                                                              \
  16 // number of threads in each dimension of a block (block will have 16x16 =
     // 256
     // threads)
#define ELEMENTS_PER_THREAD                                                    \
  2 // how many output elements each thread computes (2x2)
#define TILESIZE                                                               \
  (BLOCKSIZE * ELEMENTS_PER_THREAD) // how many output elements calculated by a
                                    // threadblock in a dimension

#define NUM_THREADS (BLOCKSIZE * BLOCKSIZE)
#define TILE_WIDTH_VECS (TILESIZE / 4) // Width of tile in float4s
#define TOTAL_VECS ((TILESIZE * TILESIZE) / 4)
#define VECS_PER_THREAD (TOTAL_VECS / NUM_THREADS)

__global__ void MatmulKernel(const float *__restrict__ a,
                             const float *__restrict__ b,
                             float *__restrict__ out, const int M, const int N,
                             const int K) {
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
  // Acts as base offsets for this thread to store into the sub-patch of shared
  // mem it is responsible for.
  int shared_A_input_tile_row = threadIdx.y * ELEMENTS_PER_THREAD;
  int shared_B_input_tile_col = threadIdx.x * ELEMENTS_PER_THREAD;

  int tid = threadIdx.y * blockDim.x +
            threadIdx.x; // linear thread id within the block
  int elements_to_load =
      (TILESIZE * TILESIZE) /
      (BLOCKSIZE * BLOCKSIZE); // how many elements each thread has to load

  float4* shared_a_vec = reinterpret_cast<float4*>(&shared_a[0][0]);
  float4* shared_b_vec = reinterpret_cast<float4*>(&shared_b[0][0]);
  
  for (int tile_k_start = 0; tile_k_start < K; tile_k_start += TILESIZE) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Iterate over the chunks this thread is responsible for
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
        
        // 1. Calculate Linear Vector Index
        // Stride is NUM_THREADS. 
        // Example: If 256 threads, Thread 0 handles vector 0, 256, 512...
        int vec_idx = tid + (v * NUM_THREADS);

        // 2. Map Linear Vector Index to 2D Tile Coordinates
        // We treat the tile as a grid of dimensions: [TILESIZE] x [TILESIZE/4]
        int row = vec_idx / TILE_WIDTH_VECS;  
        int vec_col = vec_idx % TILE_WIDTH_VECS; 
        int col = vec_col * 4; // Convert vector column back to float column

        // --- LOAD A ---
        int global_a_row = (blockIdx.y * TILESIZE) + row;
        int global_a_col = tile_k_start + col;

        if (global_a_row < M && global_a_col < K) {
             const float4* global_ptr = reinterpret_cast<const float4*>(
                 &a[global_a_row * K + global_a_col]);
             shared_a_vec[vec_idx] = __ldg(global_ptr);
        } else {
             shared_a_vec[vec_idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        // --- LOAD B ---
        int global_b_row = tile_k_start + row;
        int global_b_col = (blockIdx.x * TILESIZE) + col;

        if (global_b_row < K && global_b_col < N) {
             const float4* global_ptr = reinterpret_cast<const float4*>(
                 &b[global_b_row * N + global_b_col]);
             shared_b_vec[vec_idx] = __ldg(global_ptr);
        } else {
             shared_b_vec[vec_idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    
    __syncthreads();

    // This thread is responsible for a patch of the output tile of dim
    // ELEMENTS_PER_THREAD x ELEMENTS_PER_THREAD.
    // horizontally adjacent psums use the same row,
    // vertically adjacent psums use the same column, maybe some reuse
    // opportunity here later.
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
        int a_tile_row = shared_A_input_tile_row + i;
        int b_tile_col = shared_B_input_tile_col + j;
        for (int k = 0; k < TILESIZE; ++k) {
          partial_sums[i][j] +=
              shared_a[a_tile_row][k] * shared_b[k][b_tile_col];
        }
      }
    }
    __syncthreads(); // make sure partial sum is finished
  }
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
      if (out_row + i < M && out_col + j < N) {
        out[(out_row + i) * N + (out_col + j)] = partial_sums[i][j];
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
