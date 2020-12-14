SRCS = $(wildcard *.cc)
PROGS = $(patsubst %.cc,%,$(SRCS))
CFLAGS = -O3 -march=native -DMKL_LP64 -m64 -I${MKLROOT}/include  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl 

all: $(PROGS)

%: %.cc
	$(CXX) -o $@ $< $(CFLAGS)
clean: 
	rm -f $(PROGS)
