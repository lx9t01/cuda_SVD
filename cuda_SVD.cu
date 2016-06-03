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




__global__
void cudaMultiplyKernel(
        float* dev_P,
        float* dev_Q,
        float* dev_R1,
        int num_users,
        int num_items,
        int num_f) {
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < num_users * num_items) {
        int row = thread_idx / num_items;
        int col = thread_idx % num_items;
        for (int i = 0; i < num_f; ++i) {
            dev_R1[thread_idx] += dev_P[row * num_f + i] * dev_Q[i * num_items + col];
            // might need atomic add?
        }
        // if (thread_idx == 0) {
        //     printf("%f\n", dev_R1[thread_idx]);
        // }
        thread_idx += blockDim.x * gridDim.x;
    }
}

__global__
void cudaTrainingKernel(
        int* dev_data,
        float* dev_P,
        float* dev_Q,
        float step_size,
        float regulation,
        int num_users,
        int num_items,
        int num_f,
        int batch_size) {
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < batch_size) {
        int user = dev_data[3 * thread_idx];
        int item = dev_data[3 * thread_idx + 1];
        int rate = dev_data[3 * thread_idx + 2];
        float e = rate;
        for (int i = 0; i < num_f; ++i) {
            e -= dev_P[user * num_f + i] * dev_Q[i * num_items + item];
        }
        // printf("%f \n", e);

        for (int i = 0; i < num_f; ++i) {
            float update_row = step_size * (e * dev_Q[i * num_items + item] - regulation * dev_P[user * num_f + i]);
            float update_col = step_size * (e * dev_P[user * num_f + i] - regulation * dev_Q[i * num_items + item]);
            atomicAdd(&dev_P[user * num_f + i], update_row);
            atomicAdd(&dev_Q[i * num_items + item], update_col);
        }


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
    cudaMalloc((void**)&dev_sum, sizeof(float));
    cudaMemset(dev_sum, 0, sizeof(float));

    cudaFindRMSKernel<<<blocks, threadsPerBlock>>>(
        dev_R0,
        dev_R1,
        dev_sum, 
        num_users * num_items);
    float host_sum = -1;
    cudaMemcpy(&host_sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("dev_sum: %f\n", host_sum);
    cudaFree(dev_sum);
    return host_sum;
}


void cudaCallMultiplyKernel(const unsigned int blocks, 
    const unsigned int threadsPerBlock, 
    float* dev_P, 
    float* dev_Q, 
    float* dev_R1, 
    int num_users, 
    int num_items, 
    int num_f) {
    cudaMultiplyKernel<<<blocks, threadsPerBlock>>>(
        dev_P,
        dev_Q,
        dev_R1,
        num_users,
        num_items,
        num_f);
}

void cudaCallTrainingKernel(const unsigned int blocks, 
    const unsigned int threadsPerBlock, 
    int* dev_data, 
    float* dev_P, 
    float* dev_Q, 
    float step_size,
    float regulation,
    int num_users,
    int num_items,
    int num_f,
    int batch_size) {
    cudaTrainingKernel<<<blocks, threadsPerBlock>>>(
        dev_data,
        dev_P,
        dev_Q,
        step_size,
        regulation,
        num_users,
        num_items,
        num_f,
        batch_size);
}













