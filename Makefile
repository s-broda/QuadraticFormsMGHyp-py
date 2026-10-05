CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall
LDLIBS ?= -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
CFLAGS += -mcpu=native -fblocks
LDLIBS += -framework Accelerate
endif

# Optional Fortran comparison. FORTSRC is a directory containing
# es4mghint.f90, mexinterface.f90, and slatec.f90.
FORTSRC ?=
GF ?= gfortran
FFLAGS ?= -O3 -fopenmp -fallow-invalid-boz

.PHONY: all clean fortran

all: bench_c

bench_c: bench.c es4mgh.c es4mgh.h
	$(CC) $(CFLAGS) -o $@ bench.c es4mgh.c $(LDLIBS)

fortran:
	@test -n "$(FORTSRC)" || { echo "Set FORTSRC to the Fortran source directory"; exit 1; }
	$(MAKE) bench_fort

build/slatec.f90: $(FORTSRC)/slatec.f90
	mkdir -p build
	sed -e "s|DATA DMACH(1) / Z'0010000000000000' /|DATA DMACH(1) / 2.2250738585072014D-308 /|" \
	    -e "s|DATA DMACH(2) / Z'7FEFFFFFFFFFFFFF' /|DATA DMACH(2) / 1.7976931348623157D+308 /|" \
	    -e "s|DATA DMACH(3) / Z'3CA0000000000000' /|DATA DMACH(3) / 1.1102230246251565D-16 /|" \
	    -e "s|DATA DMACH(4) / Z'3CB0000000000000' /|DATA DMACH(4) / 2.2204460492503131D-16 /|" \
	    -e "s|DATA DMACH(5) / Z'3FD34413509F79FF' /|DATA DMACH(5) / 0.301029995663981195D0 /|" \
	    $< > $@

build/slatec.o: build/slatec.f90
	$(GF) $(FFLAGS) -c $< -o $@

build/mexinterface.o: $(FORTSRC)/mexinterface.f90
	mkdir -p build
	$(GF) $(FFLAGS) -c $< -o $@

build/es4mghint.o: $(FORTSRC)/es4mghint.f90 build/mexinterface.o
	$(GF) $(FFLAGS) -c $< -o $@

bench_fort: bench_fort.f90 build/slatec.o build/es4mghint.o build/mexinterface.o
	$(GF) $(FFLAGS) -o $@ bench_fort.f90 build/es4mghint.o build/slatec.o build/mexinterface.o

clean:
	rm -rf build bench_c bench_fort
