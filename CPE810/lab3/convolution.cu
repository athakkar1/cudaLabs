#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <time.h>

#include <ctime>
__global__ void convolutionKernel(uint* input, uint* mask, uint* output,
                                  uint dimX, uint dimY, uint dimK, uint bwidth);
void constantInit(uint* data, int size, uint val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}
void randomInit(uint* data, int size) {
  srand(time(NULL));
  for (int i = 0; i < size; ++i) {
    data[i] = rand() % 15;
  };
}
void CPUConv(uint* input, uint* mask, uint* output, uint dimX, uint dimY,
             uint dimK) {
  for (int i = 0; i < dimX * dimY; i++) {
    int row = i / dimX;
    int col = i % dimX;
    int start_row = row - dimK / 2;
    int start_col = col - dimK / 2;
    int sum = 0;
    for (int j = 0; j < dimK; j++) {
      for (int k = 0; k < dimK; k++) {
        int curRow = start_row + j;
        int curCol = start_col + k;
        if (curRow > -1 && curRow < dimY && curCol > -1 && curCol < dimX) {
          sum += input[curRow * dimX + curCol] * mask[j * dimK + k];
        }
      }
    }
    output[row * dimX + col] = sum;
  }
}
#define O_SIZE 24
#define MAX_MASK 9
#define B_SIZE O_SIZE + (MAX_MASK - 1)
void convolution(uint32_t dimX, uint32_t dimY, uint32_t dimK) {
  uint *input_h, *input_d, *output_h, *output_d, *mask_h, *mask_d, *test_out;
  size_t array = sizeof(uint) * dimX * dimY;
  size_t mask = sizeof(uint) * dimK * dimK;
  checkCudaErrors(cudaMallocHost((void**)&input_h, array));
  checkCudaErrors(cudaMallocHost((void**)&output_h, array));
  checkCudaErrors(cudaMallocHost((void**)&mask_h, mask));
  checkCudaErrors(cudaMallocHost((void**)&test_out, array));
  randomInit(input_h, dimX * dimY);
  // constantInit(input_h, dimX * dimY, 10);
  // constantInit(mask_h, dimK * dimK, 5);
  randomInit(mask_h, dimK * dimK);
  checkCudaErrors(cudaMalloc(&input_d, array));
  checkCudaErrors(cudaMemcpy(input_d, input_h, array, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&mask_d, mask));
  checkCudaErrors(cudaMemcpy(mask_d, mask_h, mask, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&output_d, array));
  uint32_t BLOCK_SIZE = O_SIZE + MAX_MASK - 1;
  dim3 dimBlock(B_SIZE, B_SIZE);
  dim3 dimGrid(
      (dimX + O_SIZE - 1) / O_SIZE,
      (dimY + O_SIZE - 1) / O_SIZE);  // how do you calculate these dimensions
  // invoke kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  for (int i = 0; i < 300; i++) {
    convolutionKernel<<<dimGrid, dimBlock>>>(input_d, mask_d, output_d, dimX,
                                             dimY, dimK, BLOCK_SIZE);
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 300;
  float gflops = (2 * dimX * dimY * dimK * dimK) / milliseconds / 1e6;
  printf("Performance: %f GFLOPS\n", gflops);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(
      cudaMemcpy(output_h, output_d, array, cudaMemcpyDeviceToHost));
  bool good = true;
  std::clock_t begin = std::clock();
  CPUConv(input_h, mask_h, test_out, dimX, dimY, dimK);
  std::clock_t end = std::clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  float gflopsCPU = (2 * dimX * dimY * dimK * dimK) / elapsed_secs / 1e9;
  std::cout << "CPUConv: " << gflopsCPU << " GLOPS\n";
  for (int i = 0; i < dimX * dimY; i++) {
    if (i % dimX == 0) {
      // printf("\n");
    }
    // printf("%d, ", output_h[i]);
    if (output_h[i] != test_out[i]) {
      // printf("%d, ", test_out[i]);
      good = false;
    }
  }
  if (good == false) {
    printf("Failed\n");
  } else {
    printf("Success\n");
  }
  printf("\n");
  checkCudaErrors(cudaFreeHost(input_h));
  checkCudaErrors(cudaFreeHost(mask_h));
  checkCudaErrors(cudaFreeHost(output_h));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(mask_d));
  checkCudaErrors(cudaFree(output_d));
}
__global__ void convolutionKernel(uint* input, uint* mask, uint* output,
                                  uint dimX, uint dimY, uint dimK,
                                  uint bwidth) {
  __shared__ int input_shared[B_SIZE][B_SIZE];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * O_SIZE + ty;
  int col = blockIdx.x * O_SIZE + tx;
  int row_i = row - (dimK / 2);
  int col_i = col - (dimK / 2);
  if (row_i > -1 && row_i < dimY && col_i > -1 && col_i < dimX) {
    input_shared[ty][tx] = input[row_i * dimX + col_i];
  } else {
    input_shared[ty][tx] = 0;
  }
  int sum = 0;
  __syncthreads();
  if (tx < O_SIZE && ty < O_SIZE) {
    for (int i = 0; i < dimK; i++) {
      for (int j = 0; j < dimK; j++) {
        sum += mask[i * dimK + j] * input_shared[ty + i][tx + j];
      }
    }
    if (row < dimY && col < dimX) {
      output[row * dimX + col] = sum;
    }
  }
  __syncthreads();
}
int main(int argc, char** argv) {
  int dimX, dimY, dimK;
  if (checkCmdLineFlag(argc, (const char**)argv, "dimX")) {
    dimX = getCmdLineArgumentInt(argc, (const char**)argv, "dimX");
  }
  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "dimY")) {
    dimY = getCmdLineArgumentInt(argc, (const char**)argv, "dimY");
  }
  if (checkCmdLineFlag(argc, (const char**)argv, "dimK")) {
    dimK = getCmdLineArgumentInt(argc, (const char**)argv, "dimK");
  }
  printf("%d, %d, %d\n", dimX, dimY, dimK);
  convolution(dimX, dimY, dimK);
  printf("\n");
  return 0;
}