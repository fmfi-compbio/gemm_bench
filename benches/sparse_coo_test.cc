#include "mkl.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>

using namespace std;

void go(int n_layers, int batch_size, int hidden_size) {
    int ALIGN = 64;
    int REPS = 100;
    batch_size = 1;
    hidden_size = 3;

    sparse_matrix_t cooA;
    int cooRowInd[] = {0, 1, 2};
    int cooColInd[] = {1, 2, 0};
    float cooVal[] = {1, 2, 3};

    mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, 
                            hidden_size, hidden_size,
                            3, //nnz
                            cooRowInd,
                            cooColInd,
                            cooVal);

    float data[] = {-1, 1, 2};
    float result[] = {0, 0, 0};
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                    1, cooA, descrA, SPARSE_LAYOUT_ROW_MAJOR,
                    data,
                    1,
                    1,
                    0,
                    result,
                    1);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", result[i*3+j]);
        }
        printf("\n");
    }


    mkl_sparse_destroy(cooA);

}

int main(int argc, char** argv) {
    int n_layers = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);

    for (int i = 0; i < 10; i++) {
        go(n_layers, batch, hidden_size);
    }
}
