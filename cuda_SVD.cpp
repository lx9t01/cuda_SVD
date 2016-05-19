// cuda_SVD.cpp

#include "cuda_SVD.h"
#define batch_size 1024

using Eigen::MatrixXd;
using namespace Eigen;



void readData(string str, int *output) {
    stringstream stream(str);
    int idx = 0;
    for (string component; getline(stream, component, '\t'); ++idx) {
        if (idx == 3) break;
        output[idx] = atoi(component.c_str());
    }
    // assert(component_idx == 3);
}


void decompose_CPU(istream& buffer, 
    const int batch_size, 
    const int num_users, 
    const int num_items, 
    const int num_f) {
    MatrixXf P(num_users, num_f);
    MatrixXf Q(num_f, num_items);

    MatrixXd R(num_users, num_items);
    gaussianFill(P, num_users, num_f);


}





int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("./classify <path to datafile> <number of users> <number of items> <number of dimensions f>\n");
        return -1;
    }

    const int num_users;
    num_users = atoi(argv[2]);
    const int num_items;
    num_items = atoi(argv[3]);
    const int num_f;
    num_f = atoi(argv[4]);

    ifstream infile(argv[1]);
    stringstream buffer;
    buffer << infile.rdbuf();
    decompose_CPU(buffer, batch_size, num_users, num_items, num_f);


}