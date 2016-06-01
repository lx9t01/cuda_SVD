#include <cstdio>
#include "cuda_SVD.cuh"

__global__
void cudaFindRMSKernel(
	float* dev_R0,
	float* dev_R1,
	float* dev_sum, 
	int num) {

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	while (thread_idx < num) {
		if (dev_R0[thread_idx] != 0) {
			atomicAdd(dev_sum, (dev_R0[thread_idx]-dev_R1[thread_idx])*(dev_R0[thread_idx]-dev_R1[thread_idx]));
		}
		// __syncthreads();
		// printf("%f\n", dev_sum);
		thread_idx += blockDim.x * gridDim.x;
	}

}

float cudaCallFindRMSKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock,
    float* dev_R0, 
    float* dev_R1, 
    int num_users, 
    int num_items) {

	float *dev_sum;
	cudaMalloc(&dev_sum, sizeof(float));
	cudaMemset(dev_sum, 0, sizeof(float));

	cudaFindRMSKernel<<<blocks, threadsPerBlock>>>(
		dev_R0,
		dev_R1,
		dev_sum, 
		num_users * num_items);
	float host_sum = -1;
	cudaMemcpy(&host_sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_sum);
	return host_sum;
}