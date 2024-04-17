#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <time.h>
__global__ void histogramKernel(uint* hist, uint* bins, uint inputSize,
                                uint binNum, uint binSize);
void constantInit(uint* data, int size, uint val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}
void randomInit(uint* data, int size) {
  srand(time(NULL));
  for (int i = 0; i < size; ++i) {
    data[i] = rand() % 1023;
  };
}
#define BLOCK_SIZE 512
void histogram(uint32_t binNum, uint32_t inputSize) {
  uint *hist_h, *hist_d, *bins_h, *bins_d;
  size_t hist = sizeof(uint) * inputSize;
  size_t bins = sizeof(uint) * binNum;
  uint binSize = 1024 / binNum;
  int power = 0;
  int finder = 1;
  while (binSize ^ finder) {
    power++;
    finder = finder << 1;
  }
  printf("%d\n", power);
  checkCudaErrors(cudaMallocHost((void**)&hist_h, hist));
  checkCudaErrors(cudaMallocHost((void**)&bins_h, bins));
  randomInit(hist_h, inputSize);
  // constantInit(hist_h, inputSize, 1020);
  constantInit(bins_h, binNum, 0);
  checkCudaErrors(cudaMalloc(&hist_d, hist));
  checkCudaErrors(cudaMemcpy(hist_d, hist_h, hist, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&bins_d, bins));
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((inputSize + BLOCK_SIZE - 1) /
               (BLOCK_SIZE * 128));  // how do you calculate these dimensions
  // invoke kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  for (int i = 0; i < 1; i++) {
    histogramKernel<<<dimGrid, dimBlock, 1>>>(hist_d, bins_d, inputSize, binNum,
                                              power);
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 1;
  float gflops = (1.0f * inputSize) / milliseconds / 1e6;
  printf("Performance: %f GFLOPS\n", gflops);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(cudaMemcpy(bins_h, bins_d, bins, cudaMemcpyDeviceToHost));
  for (int i = 0; i < binNum; i++) {
    printf("Bin %d: %d\n", i + 1, bins_h[i]);
  }
  printf("\n");
  checkCudaErrors(cudaFreeHost(hist_h));
  checkCudaErrors(cudaFreeHost(bins_h));
  checkCudaErrors(cudaFree(hist_d));
  checkCudaErrors(cudaFree(bins_d));
}
__global__ void histogramKernel(uint* hist, uint* bins, uint inputSize,
                                uint binNum, uint binSize) {
  __shared__ uint priv_hist[256];
  if (threadIdx.x < binNum) {
    priv_hist[threadIdx.x] = 0;
  }
  __syncthreads();
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  while (index < inputSize) {
    // uint bin = floor(float(hist[index]) / float(binSize));
    uint bin = hist[index] >> binSize;
    atomicAdd(&priv_hist[bin], 1);
    index += stride;
  }
  __syncthreads();
  if (threadIdx.x < binNum) {
    atomicAdd(&bins[threadIdx.x], priv_hist[threadIdx.x]);
  }
}
int main(int argc, char** argv) {
  int binNum, inputSize;
  if (checkCmdLineFlag(argc, (const char**)argv, "binNum")) {
    binNum = getCmdLineArgumentInt(argc, (const char**)argv, "binNum");
  }
  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "inputSize")) {
    inputSize = getCmdLineArgumentInt(argc, (const char**)argv, "inputSize");
  }
  printf("%d, %d\n", binNum, inputSize);
  if ((binNum & (binNum - 1)) == 0 && binNum <= 256 && binNum >= 4) {
    histogram(binNum, inputSize);
  } else {
    printf("binNum wrong");
  }

  printf("\n");
  return 0;
}