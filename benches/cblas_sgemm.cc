#include "mkl.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>

using namespace std;

void go(int n_layers, int batch_size, int hidden_size) {
    int ALIGN = 64;
    int REPS = 100;

    vector<float*> vecs;
    vector<float*> mats;

    for (int i = 0; i < 2; i++) {
        float *vec = (float*) aligned_alloc(ALIGN, batch_size*hidden_size*sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                vec[i*hidden_size+j] = j / 100.0 - hidden_size/2 + i / 10000.0;
            }
        }
        vecs.push_back(vec);
    }
    
    for (int i = 0; i < n_layers; i++) {
        float *mat = (float*) aligned_alloc(ALIGN, hidden_size*hidden_size*sizeof(float));
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                mat[i*hidden_size+j] = i*hidden_size+j / 100.0 - hidden_size*hidden_size/2;
            }
        }
        mats.push_back(mat);
    }

    auto start = std::chrono::system_clock::now();
    for (int r = 0; r < REPS; r++) {
        for (int i = 0; i < n_layers; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, hidden_size, hidden_size, 1.0,
                        vecs[i%2], hidden_size, mats[i], hidden_size, 0.0, vecs[(i+1)%2], hidden_size);
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    cout << "dense cblas_sgemm " << n_layers << " " << batch_size << " " << hidden_size << " ";
    cout << elapsed_seconds.count() << " " << (long long)batch_size*hidden_size*hidden_size*n_layers*REPS / elapsed_seconds.count() << endl;


    for (int i = 0; i < 2; i++) {
        free(vecs[i]);
    }
    for (int i = 0; i < n_layers; i++) {
        free(mats[i]);
    }
    
}

int main(int argc, char** argv) {
    int n_layers = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);

    for (int i = 0; i < 10; i++) {
        go(n_layers, batch, hidden_size);
    }
}
