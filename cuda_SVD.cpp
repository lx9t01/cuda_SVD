// cuda_SVD.cpp

#include "cuda_SVD.h"


using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

#define BATCH_SIZE 1000
#define PARSE_THROUGH 1


/*
 the training of CPU PQ decomposition
 because we want to terminate when the decrease of error is small enough, 
 so we have to keep all the training data stored in a vector, and randomly shuffle
 the data after each iteration.
*/
void decompose_CPU(stringstream& buffer, 
    int batch_size, 
    int num_users, 
    int num_items, 
    int num_f, 
    float step_size, 
    float regulation) {

    MatrixXf P(num_users, num_f);
    MatrixXf Q(num_f, num_items);

    MatrixXf R(num_users, num_items);
    gaussianFill(P, num_users, num_f);
    gaussianFill(Q, num_f, num_items);
    vector< vector<int> > data = vector< vector<int> > (); 
    // create a vector to store all of training data

    int review_idx = 0;
    for (string user_rate; getline(buffer, user_rate); ++review_idx) {
        int host_buffer[3];
        readData(user_rate, &host_buffer[0]);
        host_buffer[0]--; // transform 1-based data to 0-based index
        host_buffer[1]--;
        vector<int> line(begin(host_buffer), end(host_buffer));
        data.push_back(line);
        // if(host_buffer[0]>=942)host_buffer[0]=942;
        // if(host_buffer[1]>=1600)host_buffer[1]=1600;
        // cout << host_buffer[0] << ' ' << host_buffer[1] << ' ' << host_buffer[2] << endl;
        R(host_buffer[0], host_buffer[1]) = host_buffer[2]; // record the rating
    }
    // for (auto it = data.begin(); it != data.end(); ++it) {
    //     cout << (*it)[0] << " "<<(*it)[1]<<" "<<(*it)[2]<<endl;
    // }
    float RMS = 0;
    float RMS_new = 0;
    float delta = 1;
    float delta_new = 1;

    {
        MatrixXf R_1 = P * Q;
        for (int i = 0; i < num_users; ++i) {
            for (int j = 0; j < num_items; ++j) {
                if (R(i, j) != 0) {
                    RMS += (R_1(i, j) - R(i, j)) * (R_1(i, j) - R(i, j));
                }
            }
        }
        RMS /= review_idx;
        RMS = sqrt(RMS);
        cout << RMS << endl;
        RMS_new = RMS;
    }
    MatrixXf R_1; // create the updated matrix

    /*
    stop condition is the realtive decrease in error: when the ratio of 
    previous decrease in error to current decrease in error is less than 0.05
    */
    while (delta_new / delta >= 0.02) {
        cout << "stop condition: " << (delta_new / delta) << endl;
        RMS = RMS_new;
        delta = delta_new;
        int iteration = data.size() / batch_size;
        for (int i = 0; i < iteration; ++i) {
            for (int j = 0; j < batch_size; ++j) {
                vector<int> rating = data[i * batch_size + j];
                float e = rating[2] - P.row(rating[0]).dot(Q.col(rating[1]));
                // cout << i * batch_size + j << endl;
                // cout << rating[0] << " "<< rating[1]<< " " << rating[2] << endl;
                P.row(rating[0]) += step_size * (e * (Q.col(rating[1])).transpose() - regulation * P.row(rating[0]));
                Q.col(rating[1]) += step_size * (e * (P.row(rating[0])).transpose() - regulation * Q.col(rating[1]));
                // cout << step_size * (e * (Q.col(rating[1])).transpose() - regulation * P.row(rating[0])) << endl;
                // getchar();
            }
            R_1 = P * Q;
            RMS_new = 0;
            // this piece of code is used to compute root mean square
            // value of rating matrix error,
            // will be replaced with a GPU kerne;
            for (int k = 0; k < num_users; ++k) {
                for (int j = 0; j < num_items; ++j) {
                    if (R(k, j) != 0) {
                        RMS_new += (R_1(k, j) - R(k, j)) * (R_1(k, j) - R(k, j));
                    }
                }
            }
            RMS_new /= review_idx;
            RMS_new = sqrt(RMS_new);
            cout << "RMS: " << RMS_new << endl;
        }
        delta_new = RMS - RMS_new;
        cout << "delta_new: " << delta_new << endl;
        // getchar();

        if (PARSE_THROUGH) {
            break;
        } else {
            random_shuffle(data.begin(), data.end());
        }
    }
    printf("Training complete, writing result rating matrix to CSV....\n");
    writeCSV(R_1, "output_CPU.csv");
}

void decompose_GPU(stringstream& buffer, 
    int batch_size, 
    int num_users, 
    int num_items, 
    int num_f, 
    float step_size, 
    float regulation) {

    float *host_P = (float*)malloc(sizeof(float) * num_users * num_f); // host of R
    float *host_Q = (float*)malloc(sizeof(float) * num_f * num_items); // host of Q

    float *host_R = (float*)malloc(sizeof(float) * num_users * num_items); // host of R, 'correct result'

    gaussianFill(host_P, num_users, num_f);
    gaussianFill(host_Q, num_f, num_items);
    memset(host_R, 0, sizeof(float) * num_users * num_items);
    
    // vector< vector<int> > data_GPU = vector< vector<int> > (); 

    int review_idx = 0;
    const unsigned int blocks = 64;
    const unsigned int threadsPerBlock = 64;

    float *dev_P;
    cudaMalloc((void**) &dev_P, sizeof(float) * num_users * num_f);
    cudaMemcpy(dev_P, host_P, sizeof(float) * num_users * num_f, cudaMemcpyHostToDevice);

    float *dev_Q;
    cudaMalloc((void**) &dev_Q, sizeof(float) * num_f * num_items);
    cudaMemcpy(dev_Q, host_Q, sizeof(float) * num_f * num_items, cudaMemcpyHostToDevice);

    int *host_buffer = (int*) malloc(sizeof(int) * 3 * batch_size);
    int *dev_data; 
    cudaMalloc((void**) &dev_data, sizeof(int) * 3 * batch_size);

    for (string user_rate; getline(buffer, user_rate); ++review_idx) {
        int idx = review_idx % batch_size;
        readData(user_rate, &host_buffer[3 * idx]);
        host_buffer[3 * idx]--; // the user and item are 1 indexed, in the matrix it should be 0 indexed
        host_buffer[3 * idx + 1]--;
        if (idx == batch_size - 1) { // the buffer is full
            // cout << idx << " " << host_buffer[3 * idx] << " " << host_buffer[3 * idx + 1] << " " << host_buffer[3 * idx + 2] << endl;

            gpuErrChk(cudaMemcpy(dev_data, host_buffer, sizeof(int) * 3 * batch_size, cudaMemcpyHostToDevice));
            cudaCallTrainingKernel(blocks, 
                    threadsPerBlock, 
                    dev_data, 
                    dev_P, 
                    dev_Q, 
                    step_size,
                    regulation,
                    num_users,
                    num_items,
                    num_f,
                    batch_size);
            gpuErrChk(cudaMemcpy(host_P, dev_P, sizeof(float) * num_users * num_f, cudaMemcpyDeviceToHost));
            // for (int i = 0; i < num_f; ++i) {
            //     cout << host_P[i] << " " << endl;
            // }
            
        }
        host_R[ host_buffer[3 * idx] * num_items + host_buffer[3 * idx + 1] ] = host_buffer[3 * idx + 2]; // read in the R data
    }

    
    // the correct R matrix
    float *dev_R0;
    gpuErrChk(cudaMalloc((void**) &dev_R0, sizeof(float) * num_users * num_items)); 
    gpuErrChk(cudaMemcpy(dev_R0, host_R, sizeof(float) * num_users * num_items, cudaMemcpyHostToDevice));

    float *dev_R1;
    gpuErrChk(cudaMalloc((void**) &dev_R1, sizeof(float) * num_users * num_items));
    gpuErrChk(cudaMemset(dev_R1, 0, sizeof(float) * num_users * num_items));

    float RMS = 0;
    // float RMS_new = 0;
    // float delta = 1;
    // float delta_new = 1;
    // multiply P and Q in GPU, results stored in dev_R1
    cudaCallMultiplyKernel(blocks, 
        threadsPerBlock, 
        dev_P, 
        dev_Q, 
        dev_R1, 
        num_users, 
        num_items, 
        num_f);
    // compare the RMS loss between dev_R0 and dev_R1
    RMS = cudaCallFindRMSKernel(blocks, 
        threadsPerBlock, 
        dev_R0, 
        dev_R1, 
        num_users, 
        num_items);
    cout << "GPU SUM of RMS: " << RMS << endl;
    RMS /= review_idx;
    RMS = sqrt(RMS);
    cout << "GPU RMS: " << RMS << endl;

    float *host_R_1 = (float*)malloc(sizeof(float) * num_users * num_items); 
    gpuErrChk(cudaMemcpy(host_R_1, dev_R1, sizeof(float) * num_users * num_items, cudaMemcpyDeviceToHost));
    printf("Training complete in GPU, writing result rating matrix to CSV....\n");
    writeCSV(host_R_1, num_users, num_items, "output_GPU.csv");

    free(host_P);
    free(host_Q);
    free(host_R);

    cudaFree(dev_P);
    cudaFree(dev_Q);
    cudaFree(dev_R0);
    cudaFree(dev_R1);
    free(host_R_1);
    free(host_buffer);
    cudaFree(dev_data);
    /*
        

        while (delta_new / delta >= 0.02) {
            cout << "stop condition GPU: " << (delta_new / delta) << endl;
            RMS = RMS_new;
            delta = delta_new;
            int iteration = data_GPU.size() / batch_size;

            for (int i = 0; i < iteration; ++i) {
                // copy batches of training data into GPU
                // vector<int> temp = data_GPU[i * batch_size];
                // cout<< temp[0] << " " << temp[1] << " " << temp[2] <<endl;
                gpuErrChk(cudaMemcpy(dev_data, &(data_GPU[i * batch_size])[0], sizeof(int) * 3 * batch_size, cudaMemcpyHostToDevice));
            
                // test
                int* test0 = (int*)malloc(sizeof(int) * 3 * batch_size);
                gpuErrChk(cudaMemcpy(test0, dev_data, sizeof(int) * 3 * batch_size, cudaMemcpyDeviceToHost));
                for (int j = 0; j < batch_size; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        printf("%d ", test0[3 * j + k]);
                    }
                    printf("\n");
                }

                
                
                getchar();
            }

            cudaMemcpy(host_P, dev_P, sizeof(float) * num_users * num_f, cudaMemcpyDeviceToHost);
            printf("host_P: %f\n", host_P[0]);

            // call R_1 = P * Q after training in a batch
            cudaCallMultiplyKernel(blocks, 
                threadsPerBlock, 
                dev_P, 
                dev_Q, 
                dev_R1, 
                num_users, 
                num_items, 
                num_f);

            float *temp = (float*)malloc(sizeof(float) * num_users * num_items);

            cudaMemcpy(temp, dev_R1, sizeof(float) * num_users * num_items, cudaMemcpyDeviceToHost);
            printf("calculated: %f\n", temp[0]);
            printf("host_R correct: %f\n", host_R[0]);

            RMS_new = cudaCallFindRMSKernel(blocks, 
                threadsPerBlock, 
                dev_R0, 
                dev_R1, 
                num_users, 
                num_items);

            cout << "GPU SUM of RMS_new: " << RMS_new << endl;
            RMS_new /= review_idx;
            RMS_new = sqrt(RMS_new);
            cout << "GPU RMS_new: " << RMS_new << endl;

            delta_new = RMS - RMS_new;
            cout << "delta_new: " << delta_new << endl;

            getchar();
            random_shuffle(data_GPU.begin(), data_GPU.end());
        }
        float *host_R_1 = (float*)malloc(sizeof(float) * num_users * num_items); 
        cudaMemcpy(host_R_1, dev_R1, sizeof(float) * num_users * num_items, cudaMemcpyDeviceToHost);

        printf("Training complete in GPU, writing result rating matrix to CSV....\n");
        writeCSV(host_R_1, num_users, num_items, "output_GPU.csv");


        free(host_P);
        free(host_Q);
        free(host_R);

        cudaFree(dev_P);
        cudaFree(dev_Q);
        cudaFree(dev_R0);
        cudaFree(dev_R1);
        free(host_R_1);
    */
}




int main(int argc, char* argv[]) {
    int num_users;
    int num_items;
    int num_f;
    if (argc == 3) {
        num_users = 943;
        num_items = 1682;
        num_f = 30;
    } else if (argc == 6){
        num_users = atoi(argv[2]);
        num_items = atoi(argv[3]);
        num_f = atoi(argv[4]);
    } else {
        printf("./classify <path to training datafile> <patht to test data file> \
            (<number of users> <number of items> <number of dimensions f>)\n");
        return -1;
    }
    const float gamma = 0.0001;
    const float lamda = 0.00005;

    // CPU decomposition
    float time_initial, time_final;
    time_initial = clock();

    ifstream infile_t(argv[1]); // the training data
    ifstream infile_v(argv[2]); // the testing data

    stringstream buffer1, buffer2;
    buffer1 << infile_t.rdbuf();
    buffer2 << infile_v.rdbuf();

    decompose_CPU(buffer1, BATCH_SIZE, num_users, num_items, num_f, gamma, lamda);

    time_final = clock();
    printf("Total time to run classify on CPU: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);
    // end of CPU decomposition

    // GPU decomposition

    decompose_GPU(buffer1, BATCH_SIZE, num_users, num_items, num_f, gamma, lamda);
    // end of GPU decomposition






















    return 1;
}