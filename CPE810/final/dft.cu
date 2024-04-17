
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define BUFFER_SIZE 65536
#define MASK_LEN 115
#define OUTPUT_SIZE 796
#define FFT_SIZE 65536 / 2
#define B_SIZE OUTPUT_SIZE + (MASK_LEN - 1)
#define BLOCK_SIZE 512
int *buffer, *buffer_out;
__global__ void DFT_kernel(int* input, cuFloatComplex* output);
__global__ void IDFT_kernel(cuFloatComplex* input, int* output);
void randomInit(int* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = int(1000 * sin(3 * M_PI / 4 * i));
    // printf("%d, ", data[i]);
  };
}
void dft() {
  cuFloatComplex* dftinput_d;
  int *input_d, *output_d;
  checkCudaErrors(cudaMalloc(&input_d, BUFFER_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemcpy(input_d, buffer, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&output_d, BUFFER_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc(&dftinput_d, FFT_SIZE * sizeof(cuFloatComplex)));
  dim3 dimBlock_input(BLOCK_SIZE);
  dim3 dimGrid(FFT_SIZE / BLOCK_SIZE);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  DFT_kernel<<<dimGrid, dimBlock_input>>>(input_d, dftinput_d);
  dim3 dimGridInverse(BUFFER_SIZE / BLOCK_SIZE);
  IDFT_kernel<<<dimGridInverse, dimBlock_input>>>(dftinput_d, output_d);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fprintf(stderr, "Milliseconds for Kernel: %f", milliseconds);
  checkCudaErrors(cudaMemcpy(buffer_out, output_d, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(dftinput_d));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(output_d));
}

__global__ void DFT_kernel(int* input, cuFloatComplex* output) {
  __shared__ int input_shared[BLOCK_SIZE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float c = -2 * M_PI * k / BUFFER_SIZE;
  cuFloatComplex sum = make_cuFloatComplex(0, 0);
  for (int p = 0; p < BUFFER_SIZE / BLOCK_SIZE; p++) {
    input_shared[threadIdx.x] = input[p * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float imag, real;
      sincosf(c * ((p * BLOCK_SIZE) + n), &imag, &real);
      sum = cuCaddf(sum, make_cuFloatComplex(real * input_shared[n],
                                             imag * input_shared[n]));
    }
    __syncthreads();
  }
  output[k] = sum;
  // test[k] = sqrt((output[k].x * output[k].x) + (output[k].y * output[k].y));
#if 0 
if ((k * 44100) / BUFFER_SIZE < 6174) { output[k] = sum; }
  else {
    output[k] = make_cuFloatComplex(0, 0);
  }
#endif
}
__global__ void IDFT_kernel(cuFloatComplex* input, int* output) {
  __shared__ cuFloatComplex input_shared[BLOCK_SIZE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float c = 2 * M_PI * k / FFT_SIZE;
  int sum = 0;
  for (int p = 0; p < FFT_SIZE / BLOCK_SIZE; p++) {
    input_shared[threadIdx.x] = input[p * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float imag, real;
      sincosf(c * ((p * BLOCK_SIZE) + n), &imag, &real);
      sum = sum + cuCmulf(input_shared[n], make_cuFloatComplex(real, imag)).x;
    }
    __syncthreads();
  }
  output[k] = int(sum / FFT_SIZE);
}
int main() {
  size_t buffer_size = BUFFER_SIZE * sizeof(int);
  checkCudaErrors(cudaMallocHost((void**)&buffer, buffer_size));
  checkCudaErrors(cudaMallocHost((void**)&buffer_out, buffer_size));
  randomInit(buffer, BUFFER_SIZE);
  dft();
  std::ofstream file("output.csv");

  for (int i = 0; i < BUFFER_SIZE; i++) {
    file << i << "," << buffer_out[i] << "\n";
    printf("%d, ", buffer_out[i]);
  }

  file.close();
  checkCudaErrors(cudaFreeHost(buffer));
  checkCudaErrors(cudaFreeHost(buffer_out));
  return 0;
}
