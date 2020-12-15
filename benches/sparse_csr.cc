#include "mkl.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

void go(int n_layers, int batch_size, int hidden_size, int nnz) {
    int ALIGN = 64;
    int REPS = 100;

    vector<int*> rinds;
    vector<int*> cinds;
    vector<float*> vals;

    vector<float*> vecs;
    vector<sparse_matrix_t> mats;
    vector<sparse_matrix_t> coo_mats;


    for (int i = 0; i < 2; i++) {
        float *vec = (float*) aligned_alloc(ALIGN, batch_size*hidden_size*sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                vec[i*hidden_size+j] = j / 100.0 - hidden_size/2 + i / 10000.0;
            }
        }
        vecs.push_back(vec);
    }

    vector<pair<int, int>> all;
    for (int j = 0; j < hidden_size; j++) {
        for (int k = 0; k < hidden_size; k++) {
            all.push_back(make_pair(j, k));
        }
    }
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    for (int i = 0; i < n_layers; i++) {
        random_shuffle(all.begin(), all.end());
        sort(all.begin(), all.begin()+nnz);
        float *v = (float*) aligned_alloc(ALIGN, nnz*sizeof(float));
        int *rows = (int*) aligned_alloc(ALIGN, nnz*sizeof(int));
        int *cols = (int*) aligned_alloc(ALIGN, nnz*sizeof(int));

        for (int j = 0; j < nnz; j++) {
            rows[j] = all[j].first;
            cols[j] = all[j].second;
            v[j] = j % 7;
        }

        sparse_matrix_t coo;
        mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, 
                                hidden_size, hidden_size,
                                nnz,
                                rows,
                                cols,
                                v);
        rinds.push_back(rows);
        cinds.push_back(cols);
        vals.push_back(v);
        coo_mats.push_back(coo);

        sparse_matrix_t csr;
        mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
        mkl_sparse_optimize ( csr );
        mats.push_back(csr);
    }

    auto start = std::chrono::system_clock::now();
    for (int r = 0; r < REPS; r++) {
        for (int i = 0; i < n_layers; i++) {
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                    1, mats[i], descrA, SPARSE_LAYOUT_ROW_MAJOR,
                    vecs[i%2],
                    batch_size,
                    batch_size,
                    0,
                    vecs[(i+1)%2],
                    batch_size);

        }
    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    cout << "sparse csr " << n_layers << " " << batch_size << " " << hidden_size << " " << nnz << " ";
    cout << elapsed_seconds.count() << " " << (long long)batch_size*hidden_size*hidden_size*n_layers*REPS / elapsed_seconds.count() << endl;

    for (int i = 0; i < 2; i++) {
        free(vecs[i]);
    }
    for (int i = 0; i < n_layers; i++) {
        mkl_sparse_destroy(mats[i]);
        mkl_sparse_destroy(coo_mats[i]);
        free(vals[i]);
        free(rinds[i]);
        free(cinds[i]);
    }

}

int main(int argc, char** argv) {
    int n_layers = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nnz = atoi(argv[4]);

    for (int i = 0; i < 10; i++) {
        go(n_layers, batch, hidden_size, nnz);
    }
}
