#include "mkl.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

void go(int n_layers, int batch_size, int hidden_size, int nnz, int block_size) {
    int ALIGN = 128;
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
    for (int j = 0; j < hidden_size / block_size; j++) {
        for (int k = 0; k < hidden_size / block_size; k++) {
            all.push_back(make_pair(j, k));
        }
    }
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    for (int i = 0; i < n_layers; i++) {
        random_shuffle(all.begin(), all.end());
        sort(all.begin(), all.begin()+nnz / block_size / block_size);
        float *v = (float*) aligned_alloc(ALIGN, nnz*sizeof(float));
        int *rows = (int*) aligned_alloc(ALIGN, nnz*sizeof(int));
        int *cols = (int*) aligned_alloc(ALIGN, nnz*sizeof(int));

        for (int j = 0; j < nnz / block_size / block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                for (int k2 = 0; k2 < block_size; k2++) {
                    rows[j*block_size*block_size + k*block_size+k2] = all[j].first*block_size+k;
                    cols[j*block_size*block_size + k*block_size+k2] = all[j].second*block_size+k2;
                    v[j*block_size*block_size + k*block_size+k2] = j % 7 - k2%5 - k%3;
                }
            }
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

        sparse_matrix_t bsr;
//        mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &bsr);
        mkl_sparse_convert_bsr(coo, block_size, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsr);
        mkl_sparse_set_mm_hint (bsr, SPARSE_OPERATION_NON_TRANSPOSE, descrA, SPARSE_LAYOUT_ROW_MAJOR, batch_size, 1234567);
        mkl_sparse_optimize ( bsr );
    
        mats.push_back(bsr);
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
    cout << "blocksparse " << block_size << " " << n_layers << " " << batch_size << " " << hidden_size << " " << nnz << " ";
    cout << elapsed_seconds.count() << endl;

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
    int block_size = atoi(argv[5]);

    for (int i = 0; i < 10; i++) {
        go(n_layers, batch, hidden_size, nnz, block_size);
    }
}
