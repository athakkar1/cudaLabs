#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
__global__ void matMulKernel(float* mat1, float* mat2, float* mat3, int P,
                             int M, int N, int bsize);
void ConstantInit(float* data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}
#define BLOCK_SIZE 32

void matMul(uint32_t M, uint32_t N, uint32_t P) {
  float *mat1_d, *mat2_d, *mat3_d, *mat1_h, *mat2_h, *mat3_h;
  bool success = 1;
  size_t mat1 = sizeof(float) * M * N;
  size_t mat2 = sizeof(float) * N * P;
  size_t mat3 = sizeof(float) * M * P;
  checkCudaErrors(cudaMallocHost((void**)&mat1_h, mat1));
  checkCudaErrors(cudaMallocHost((void**)&mat2_h, mat2));
  checkCudaErrors(cudaMallocHost((void**)&mat3_h, mat3));
  ConstantInit(mat1_h, mat1, 1.0f);
  ConstantInit(mat2_h, mat2, .01f);
  checkCudaErrors(cudaMalloc(&mat1_d, mat1));
  checkCudaErrors(cudaMemcpy(mat1_d, mat1_h, mat1, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&mat2_d, mat2));
  checkCudaErrors(cudaMemcpy(mat2_d, mat2_h, mat2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&mat3_d, mat3));
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((P + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) /
                   BLOCK_SIZE);  // how do you calculate these dimensions
  // invoke kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  for (int i = 0; i < 300; i++) {
    matMulKernel<<<dimGrid, dimBlock, 1>>>(mat1_d, mat2_d, mat3_d, P, M, N,
                                           BLOCK_SIZE);
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 300;
  float gflops = (2.0f * N * M * P) / milliseconds / 1e6;
  printf("Performance: %f GFLOPS\n", gflops);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(cudaMemcpy(mat3_h, mat3_d, mat3, cudaMemcpyDeviceToHost));

  for (int i = 0; i < M * P; i++) {
    // printf("%3.2f ", mat3_h[i]);
    if (abs(mat3_h[i] - (N * .01)) > .0001) {
      printf("%3.2f ", mat3_h[i]);
      success = 0;
    }
  }
  if (success) {
    printf("Success!\n");
  } else {
    printf("Failure\n");
  }
  checkCudaErrors(cudaFreeHost(mat1_h));
  checkCudaErrors(cudaFreeHost(mat2_h));
  checkCudaErrors(cudaFreeHost(mat3_h));
  checkCudaErrors(cudaFree(mat1_d));
  checkCudaErrors(cudaFree(mat2_d));
  checkCudaErrors(cudaFree(mat3_d));
}
__global__ void matMulKernel(float* mat1, float* mat2, float* mat3, int P,
                             int M, int N, int bsize) {
  __shared__ float mat1_sub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float mat2_sub[BLOCK_SIZE][BLOCK_SIZE];
  float sum = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int p = 0; p < (N - 1) / BLOCK_SIZE + 1; p++) {
    if (row < M && p * BLOCK_SIZE + threadIdx.x < N) {
      mat1_sub[threadIdx.y][threadIdx.x] =
          mat1[row * N + p * BLOCK_SIZE + threadIdx.x];
    } else {
      mat1_sub[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (p * BLOCK_SIZE + threadIdx.y < N && col < P) {
      mat2_sub[threadIdx.y][threadIdx.x] =
          mat2[(p * BLOCK_SIZE + threadIdx.y) * P + col];
    } else {
      mat2_sub[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    if (row < M && col < P) {
      for (int i = 0; i < BLOCK_SIZE; i++) {
        sum += mat1_sub[threadIdx.y][i] * mat2_sub[i][threadIdx.x];
      }
    }
    __syncthreads();
  }
  if (row < M && col < P) {
    mat3[row * P + col] = sum;
  }
}
int main(int argc, char** argv) {
  int M, N, P;
  if (checkCmdLineFlag(argc, (const char**)argv, "hA")) {
    M = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
  } else {
    M = 32;
  }
  // M of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
    N = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
  } else {
    M = 32;
  }
  if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
    P = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
  } else {
    P = 32;
  }
  printf("%d, %d, %d\n", M, N, P);

  matMul(M, N, P);

  printf("\n");
  return 0;
}