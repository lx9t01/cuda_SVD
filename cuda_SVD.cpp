// cuda_SVD.cpp

#include "cuda_SVD.h"


using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

#define BATCH_SIZE 1000


void readData(string str, int *output) {
    stringstream stream(str);
    int idx = 0;
    for (string component; getline(stream, component, '\t'); ++idx) {
        if (idx == 3) break;
        output[idx] = atoi(component.c_str());
    }
    // assert(component_idx == 3);
}


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
    // cout << P << endl;
    // cout << Q << endl;
    vector< vector<int> > data = vector< vector<int> > ();

    int review_idx = 0;
    for (string user_rate; getline(buffer, user_rate); ++review_idx) {
        int host_buffer[3];
        readData(user_rate, &host_buffer[0]);
        host_buffer[0]--;
        host_buffer[1]--;
        vector<int> line(begin(host_buffer), end(host_buffer));
        data.push_back(line);
        // if(host_buffer[0]>=942)host_buffer[0]=942;
        // if(host_buffer[1]>=1600)host_buffer[1]=1600;
        // cout << host_buffer[0] << ' ' << host_buffer[1] << ' ' << host_buffer[2] << endl;
        R(host_buffer[0], host_buffer[1]) = host_buffer[2];
    }
    // for (auto it = data.begin(); it != data.end(); ++it) {
    //     cout << (*it)[0] << " "<<(*it)[1]<<" "<<(*it)[2]<<endl;
    // }
    float RMS = 0;
    float RMS_new = 0;
    {
        MatrixXf R_1 = P * Q;
        for (int i = 0; i < num_users; ++i) {
            for (int j = 0; j < num_items; ++j) {
                if (R(i, j) != 0) {
                    RMS += (R_1(i, j) - R(i, j)) * (R_1(i, j) - R(i, j));
                // cout<< R_1(i,j)<<endl;
                }
            }
        }
        RMS /= review_idx;
        cout << RMS << endl;
        RMS_new = RMS;
    }
    while (abs(RMS_new / RMS) >= 1e-3) {
        RMS = RMS_new;
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
            MatrixXf R_1 = P * Q;
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
            
        }
        // getchar();
        cout << RMS_new << endl;
        random_shuffle(data.begin(), data.end());
    }
    // cout << P * Q << endl;
    // {
        
    //     R(host_buffer[0], host_buffer[1]) = host_buffer[2];
    //     float e = R(host_buffer[0], host_buffer[1]) - P.row(host_buffer[0]).dot(Q.col(host_buffer[1]));
    //     P.row(host_buffer[0]) += step_size * (e * (Q.col(host_buffer[1])).transpose() - regulation * P.row(host_buffer[0]));
    //     Q.col(host_buffer[1]) += step_size * (e * (P.row(host_buffer[0])).transpose() - regulation * Q.col(host_buffer[1]));
    // }

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
        printf("./classify <path to training datafile> <patht to validation data file> \
            (<number of users> <number of items> <number of dimensions f>)\n");
        return -1;
    }
    const float gamma = 0.005;
    const float lamda = 0.001;

    ifstream infile_t(argv[1]);
    ifstream infile_v(argv[2]);
    stringstream buffer1, buffer2;
    buffer1 << infile_t.rdbuf();
    buffer2 << infile_v.rdbuf();
    decompose_CPU(buffer1, BATCH_SIZE, num_users, num_items, num_f, gamma, lamda);

    return 1;
}