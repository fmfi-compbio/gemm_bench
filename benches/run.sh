rm -f results

for nl in 6 12; do 
  for b in 1 4 16 64 256 1024; do
    for nnz in 64 128 256 512; do
      ./sparse_coo $nl $b 32 $nnz >>results;
      ./sparse_csr $nl $b 32 $nnz >>results;
    done
    for nnz in 128 256 512 1024 2048; do
      ./sparse_coo $nl $b 64 $nnz >>results;
      ./sparse_csr $nl $b 64 $nnz >>results;
    done
    for nnz in 256 512 1024 2048 4096 8192; do
      ./sparse_coo $nl $b 128 $nnz >>results;
      ./sparse_csr $nl $b 128 $nnz >>results;
    done
    for nnz in 512 1024 2048 4096 8192 16384 32768; do
      ./sparse_coo $nl $b 256 $nnz >>results;
      ./sparse_csr $nl $b 256 $nnz >>results;
    done
    for nnz in 1024 2048 4096 8192 16384 32768 65536 131072; do
      ./sparse_coo $nl $b 512 $nnz >>results;
      ./sparse_csr $nl $b 512 $nnz >>results;
    done
    ./cblas_sgemm $nl $b 32 >>results;
    ./cblas_sgemm $nl $b 64 >>results;
    ./cblas_sgemm $nl $b 128 >>results;
    ./cblas_sgemm $nl $b 256 >>results;
    ./cblas_sgemm $nl $b 512 >>results;
  done
done

