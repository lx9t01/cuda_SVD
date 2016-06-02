#ifndef CUDA_SVD_CUH
#define CUDA_SVD_CUH



float cudaCallFindRMSKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock,
    float* dev_R0, 
    float* dev_R1, 
    int num_users, 
    int num_items);

void cudaCallMultiplyKernel(const unsigned int blocks, 
    const unsigned int threadsPerBlock, 
    float* dev_P, 
    float* dev_Q, 
    float* dev_R1, 
    int num_users, 
    int num_items, 
    int num_f);

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
    int batch_size);

#endif