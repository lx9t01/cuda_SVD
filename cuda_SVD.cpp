// cuda_SVD.cpp

#include "cuda_SVD.h"


using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

#define BATCH_SIZE 1000


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

    MatrixXd R(num_users, num_items);
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
    while (delta_new / delta >= 0.05) {
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
            for (int i = 0; i < num_users; ++i) {
                for (int j = 0; j < num_items; ++j) {
                    if (R(i, j) != 0) {
                        RMS_new += (R_1(i, j) - R(i, j)) * (R_1(i, j) - R(i, j));
                    // cout<< R_1(i,j)<<endl;
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
        random_shuffle(data.begin(), data.end());
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

    MatrixXf P(num_users, num_f);
    MatrixXf Q(num_f, num_items);
    MatrixXd R(num_users, num_items);
    gaussianFill(P, num_users, num_f);
    gaussianFill(Q, num_f, num_items);
    
    vector< vector<int> > data_GPU = vector< vector<int> > (); 
    // create a vector to store all of training data

    int review_idx = 0;
    for (string user_rate; getline(buffer, user_rate); ++review_idx) {
        int host_buffer[3];
        readData(user_rate, &host_buffer[0]);
        host_buffer[0]--; // transform 1-based data to 0-based index
        host_buffer[1]--;
        vector<int> line(begin(host_buffer), end(host_buffer));
        data_GPU.push_back(line);
        R(host_buffer[0], host_buffer[1]) = host_buffer[2]; // record the rating
    }

    float RMS = 0;
    float RMS_new = 0;
    float delta = 1;
    float delta_new = 1;

    // transform the code below to a find_RMS kernel in GPU
    /*
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
    */
    const unsigned int blocks = 64;
    const unsigned int threadsPerBlock = 64;

    {
        MatrixXf R_1 = P * Q;
        float *R0 = R.data(); // host data of R
        float *R1 = R_1.data();
        float *dev_R0;
        float *dev_R1;

        cudaMalloc((void**) &dev_R0, sizeof(float) * num_users * num_items);
        cudaMalloc((void**) &dev_R1, sizeof(float) * num_users * num_items);
        cudaMemcpy(dev_R0, R0, sizeof(float) * num_users * num_items);
        cudaMemcpy(dev_R1, R1, sizeof(float) * num_items * num_items);
        RMS = cudaCallFindRMSKernel(blocks, threadsPerBlock, dev_R0, dev_R1, num_users, num_items);
        RMS /= review_idx;
        RMS = sqrt(RMS);
        cout << "GPU RMS: " << RMS << endl;
        RMS_new = RMS;

        while (delta_new / delta >= 0.05) {
            cout << "stop condition GPU: " << (delta_new / delta) << endl;
            RMS = RMS_new;
            delta = delta_new;
            int iteration = data_GPU.size() / batch_size;

        }












        cudaFree(dev_R0);
        cudaFree(dev_R1);
    }
    





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
    const float gamma = 0.01;
    const float lamda = 0.005;

    // CPU decomposition
    float time_initial, time_final;
    time_initial = clock();

    ifstream infile_t(argv[1]); // the training data
    ifstream infile_v(argv[2]); // the testing data

    stringstream buffer1, buffer2;
    buffer1 << infile_t.rdbuf();
    buffer2 << infile_v.rdbuf();

    // decompose_CPU(buffer1, BATCH_SIZE, num_users, num_items, num_f, gamma, lamda);

    time_final = clock();
    printf("Total time to run classify on CPU: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);
    // end of CPU decomposition

    // GPU decomposition

    decompose_GPU(buffer1, BATCH_SIZE, num_users, num_items, num_f, gamma, lamda);
    // end of GPU decomposition






















    return 1;
}