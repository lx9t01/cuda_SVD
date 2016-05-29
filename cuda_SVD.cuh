#ifndef CUDA_SVD_CUH
#define CUDA_SVD_CUH

// float cuda_SVD(
//     int *data)


float cudaCallFindRMSKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock,
    float* dev_R0, 
    float* dev_R1, 
    int num_users, 
    int num_items);

#endif