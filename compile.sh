export MKLROOT=/opt/intel/mkl

g++ test.cc -O3 -march=native -DMKL_ILP64 -m64 -I${MKLROOT}/include  -Wl,--start-group
${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a
${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

