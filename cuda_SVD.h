// cuda_SVD.h

#ifndef _INCL_GUARD

#define _INCL_GUARD
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <iterator>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
using namespace std;
/*
gpuErrChk
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
/*
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}
*/
// Fills output with standard normal data
void gaussianFill(MatrixXf &output, int size_row, int size_col) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size_row; ++i) {
        for (int j = 0; j < size_col; ++j) {
            output(i, j) = distribution(generator);
        }
    }
}


void decompose_CPU(stringstream& buffer, 
    int batch_size, 
    int num_users, 
    int num_items, 
    int num_f, 
    float step_size, 
    float regualtion);


#endif
