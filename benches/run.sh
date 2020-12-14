rm -f results

./cblas_sgemm 6 1 32 >>results
./cblas_sgemm 6 8 32 >>results
./cblas_sgemm 6 64 32 >>results
./cblas_sgemm 6 1 64 >>results
./cblas_sgemm 6 8 64 >>results
./cblas_sgemm 6 64 64 >>results
./cblas_sgemm 6 1 128 >>results
./cblas_sgemm 6 8 128 >>results
./cblas_sgemm 6 64 128 >>results
./cblas_sgemm 6 1 256 >>results
./cblas_sgemm 6 8 256 >>results
./cblas_sgemm 6 64 256 >>results
./cblas_sgemm 6 1 256 >>results
./cblas_sgemm 6 8 256 >>results
./cblas_sgemm 6 64 256 >>results
./cblas_sgemm 12 1 32 >>results
./cblas_sgemm 12 8 32 >>results
./cblas_sgemm 12 64 32 >>results
./cblas_sgemm 12 1 64 >>results
./cblas_sgemm 12 8 64 >>results
./cblas_sgemm 12 64 64 >>results
./cblas_sgemm 12 1 128 >>results
./cblas_sgemm 12 8 128 >>results
./cblas_sgemm 12 64 128 >>results
./cblas_sgemm 12 1 256 >>results
./cblas_sgemm 12 8 256 >>results
./cblas_sgemm 12 64 256 >>results
./cblas_sgemm 12 1 256 >>results
./cblas_sgemm 12 8 256 >>results
./cblas_sgemm 12 64 256 >>results

