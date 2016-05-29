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

#include <cuda_runtime.h>
#include "cuda_SVD.cuh"

using namespace Eigen;
using namespace std;


/*
gpuErrChk
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/

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


// Fills output with standard normal data
// input:  output (P or Q), dimension of row, dimension of column
// output: output matrix (P or Q)
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


// write the output full rating matrix to a csv file
void writeCSV(MatrixXf R, string filename) {
    int r = R.rows();
    int c = R.cols();
    ofstream outputfile;
    outputfile.open(filename);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            outputfile << R(i,j) << ",";
        }
        outputfile << "\n";
    }
    outputfile.close();
}


// read one line of the training data
// input:  str
// output: output = {userID, itemID, rating}
void readData(string str, int *output) {
    stringstream stream(str);
    int idx = 0;
    for (string component; getline(stream, component, '\t'); ++idx) {
        if (idx == 3) break;
        output[idx] = atoi(component.c_str());
    }
    // assert(component_idx == 3);
}

// decompose the matrix with CPU
/*input:  buffer: from the training data file
          batch_size: read a batch from the input file
          num_users: the total number of users
          num_items: the total number of items (used to define matrix dimension)
          num_f : the dimension of latent factors
          step_size: training step size, set to 0.01
          regulation : regulation term set to prevent overfitting, set to 0.005
output:   void; the final trained model of rating matrix will be writteninto a CSV file
*/  
void decompose_CPU(stringstream& buffer, 
    int batch_size, 
    int num_users, 
    int num_items, 
    int num_f, 
    float step_size, 
    float regualtion);

// decompose the matrix with GPU (as an interface between stream, same as CPU)
/*input:  buffer: from the training data file
          batch_size: read a batch from the input file
          num_users: the total number of users
          num_items: the total number of items (used to define matrix dimension)
          num_f : the dimension of latent factors
          step_size: training step size, set to 0.01
          regulation : regulation term set to prevent overfitting, set to 0.005
output:   void; the final trained model of rating matrix will be writteninto a CSV file
*/  
void decompose_GPU(stringstream& buffer,
    int batch_size,
    int num_users,
    int num_items,
    int num_f,
    float step_size,
    float regulation);

#endif
