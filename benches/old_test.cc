#include "mkl.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

int main() {
    int ALIGN = 64;
    for (int big_iter = 0; big_iter < 10; big_iter++) {
        MKL_INT m, n, k;
        m = 16;
        n = 64*3;
        k = 64;

        float *vec = (float*) aligned_alloc(ALIGN, m*k*sizeof(float));
        float *mat = (float*) aligned_alloc(ALIGN, n*k*sizeof(float));
        float *mat_trans = (float*) aligned_alloc(ALIGN, n*k*sizeof(float));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                vec[i*k+j] = j / 100.0 - k/2 + i / 10000.0;
//                printf("%f ", vec[i*k+j]);
            }
        }
//        printf("\n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                mat[i*n+j] = i*n+j / 100.0 - n*k/2;
                mat_trans[j*k+i] = i*n+j / 100.0 - n*k/2;
            }
        }


        float *out = (float*) aligned_alloc(ALIGN, m*n*sizeof(float));
        for (int i = 0; i < m*n; i++) out[i] = 0;

        // baseline test (direct blas call)

        for (int it = 0; it < 2; it++) {
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < 10000; i++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                            vec, k, mat, n, 0.0, out, n);
            }
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            cout << "direct  " << out[0] << " " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
        }

        for (int it = 0; it < 2; it++) {
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < 10000; i++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
                            vec, k, mat_trans, k, 0.0, out, n);
            }
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            cout << "directT " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
        }
        // jit test

        {
            void *jitter;
            mkl_cblas_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
                m, n, k, 1.0, k, n, 0.0, n);

            auto sgemm2 = mkl_jit_get_sgemm_ptr(jitter);

            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    sgemm2(jitter, vec, mat, out);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "jit     " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        }
       
        {
            void *jitter;
            mkl_cblas_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS,
                m, n, k, 1.0, k, k, 0.0, n);

            auto sgemm2 = mkl_jit_get_sgemm_ptr(jitter);

            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    sgemm2(jitter, vec, mat_trans, out);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "jitT    " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        }

        // pack test
       
        {
            size_t Ap_size = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);

            float* packed_mat = (float*) mkl_malloc(Ap_size, 64);

            cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans,
                             m, n, k, 1.0, mat, n, packed_mat);
            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                            m, n, k, vec, k, packed_mat, n, 0.0, out, n);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "pack    " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        }

        {
            size_t Ap_size = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);

            float* packed_mat = (float*) mkl_malloc(Ap_size, 64);

            cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasTrans,
                             m, n, k, 1.0, mat_trans, k, packed_mat);
            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                            m, n, k, vec, k, packed_mat, n, 0.0, out, n);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "packT   " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        }

        // int8 tests

        {
            char *vec = (char*) aligned_alloc(ALIGN, m*k*sizeof(char));
            unsigned char *mat = (unsigned char*) aligned_alloc(ALIGN, n*k*sizeof(unsigned char));
            unsigned char *mat_trans = (unsigned char*) aligned_alloc(ALIGN, n*k*sizeof(unsigned char));

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    vec[i*k+j] = j;
                }
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    mat[i*n+j] = i*n+j / 20.0;
                    mat_trans[j*k+i] = i*n+j / 20.0;
                }
            }

            MKL_INT32 *out = (MKL_INT32*) aligned_alloc(ALIGN, m*n*sizeof(MKL_INT32));
            for (int i = 0; i < m*n; i++) out[i] = 0;

            MKL_INT32 oc = 0;

        
            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            CblasFixOffset, m, n, k,
                            1.0, vec, k, 0, mat, n, 0, 0.0, out, n, &oc);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "i8      " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }

            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            CblasFixOffset, m, n, k,
                            1.0, vec, k, 0, mat_trans, k, 0, 0.0, out, n, &oc);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "i8T     " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        
            {
                size_t Ap_size = cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix, m, n, k);

                unsigned char* packed_mat = (unsigned char*) mkl_malloc(Ap_size, 64);

                cblas_gemm_s8u8s32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans,
                                 m, n, k, mat, n, packed_mat);
                for (int it = 0; it < 2; it++) {
                    auto start = std::chrono::system_clock::now();
                    for (int i = 0; i < 10000; i++) {
                        cblas_gemm_s8u8s32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                CblasFixOffset, 
                                m, n, k, 1.0, vec, k, 0, packed_mat, n, 0, 0.0, out, n, &oc);
                    }
                    auto end = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = end-start;
                    cout << "packi8  " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
                }
            }
            {
                size_t Ap_size = cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix, m, n, k);

                unsigned char* packed_mat = (unsigned char*) mkl_malloc(Ap_size, 64);

                cblas_gemm_s8u8s32_pack(CblasRowMajor, CblasBMatrix, CblasTrans,
                                 m, n, k, mat_trans, k, packed_mat);
                for (int it = 0; it < 2; it++) {
                    auto start = std::chrono::system_clock::now();
                    for (int i = 0; i < 10000; i++) {
                        cblas_gemm_s8u8s32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                CblasFixOffset, 
                                m, n, k, 1.0, vec, k, 0, packed_mat, n, 0, 0.0, out, n, &oc);
                    }
                    auto end = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = end-start;
                    cout << "packi8T " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
                }
            }
        
        }
        // int16 tests

        {
            MKL_INT16 *vec = (MKL_INT16*) aligned_alloc(ALIGN, m*k*sizeof(MKL_INT16));
            MKL_INT16 *mat = (MKL_INT16*) aligned_alloc(ALIGN, n*k*sizeof(MKL_INT16));
            MKL_INT16 *mat_trans = (MKL_INT16*) aligned_alloc(ALIGN, n*k*sizeof(MKL_INT16));

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    vec[i*k+j] = j;
                }
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    mat[i*n+j] = i*n+j / 20.0;
                    mat_trans[j*k+i] = i*n+j / 20.0;
                }
            }

            MKL_INT32 *out = (MKL_INT32*) aligned_alloc(ALIGN, m*n*sizeof(MKL_INT32));
            for (int i = 0; i < m*n; i++) out[i] = 0;

            MKL_INT32 oc = 0;

        
            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            CblasFixOffset, m, n, k,
                            1.0, vec, k, 0, mat, n, 0, 0.0, out, n, &oc);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "i16      " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }

            for (int it = 0; it < 2; it++) {
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < 10000; i++) {
                    cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans,
                            CblasFixOffset, m, n, k,
                            1.0, vec, k, 0, mat_trans, k, 0, 0.0, out, n, &oc);
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                cout << "i16T     " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
            }
        
            {
                size_t Ap_size = cblas_gemm_s16s16s32_pack_get_size(CblasBMatrix, m, n, k);

                MKL_INT16* packed_mat = (MKL_INT16*) mkl_malloc(Ap_size, 64);

                cblas_gemm_s16s16s32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans,
                                 m, n, k, mat, n, packed_mat);
                for (int it = 0; it < 2; it++) {
                    auto start = std::chrono::system_clock::now();
                    for (int i = 0; i < 10000; i++) {
                        cblas_gemm_s16s16s32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                CblasFixOffset, 
                                m, n, k, 1.0, vec, k, 0, packed_mat, n, 0, 0.0, out, n, &oc);
                    }
                    auto end = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = end-start;
                    cout << "packi16  " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
                }
            }
            {
                size_t Ap_size = cblas_gemm_s16s16s32_pack_get_size(CblasBMatrix, m, n, k);

                MKL_INT16* packed_mat = (MKL_INT16*) mkl_malloc(Ap_size, 64);

                cblas_gemm_s16s16s32_pack(CblasRowMajor, CblasBMatrix, CblasTrans,
                                 m, n, k, mat_trans, k, packed_mat);
                for (int it = 0; it < 2; it++) {
                    auto start = std::chrono::system_clock::now();
                    for (int i = 0; i < 10000; i++) {
                        cblas_gemm_s16s16s32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
                                CblasFixOffset, 
                                m, n, k, 1.0, vec, k, 0, packed_mat, n, 0, 0.0, out, n, &oc);
                    }
                    auto end = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = end-start;
                    cout << "packi16T " << out[n-2] << " " << elapsed_seconds.count() << " " << m*n*k*2*10000 / elapsed_seconds.count() << endl;
                }
            }
        
        }        
        printf("\n");
    }
}
