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

#endif