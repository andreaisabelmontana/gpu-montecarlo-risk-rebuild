# Multi-backend Monte Carlo risk engine.
#   make            -> serial + openmp (CPU; builds anywhere with gcc)
#   make cuda       -> GPU backend (REQUIRES nvcc + an NVIDIA GPU)
#   make bench      -> build CPU backends and run the benchmark suite
#   make clean

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O3 -march=native -ffast-math -Wall -Wextra -Iinclude
LDLIBS  ?= -lm
NVFLAGS ?= -O3 -Iinclude

COMMON  := src/common.c

# Append .exe on Windows so the same Makefile works on both platforms.
ifeq ($(OS),Windows_NT)
  EXT := .exe
else
  EXT :=
endif

.PHONY: all cpu cuda bench clean
all: cpu

cpu: mc_serial$(EXT) mc_openmp$(EXT)

mc_serial$(EXT): src/serial.c $(COMMON)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

mc_openmp$(EXT): src/openmp.c $(COMMON)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDLIBS)

# GPU target — only works where the CUDA toolkit is installed.
cuda: mc_cuda$(EXT)
mc_cuda$(EXT): src/cuda.cu $(COMMON)
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcurand

bench: cpu
	python scripts/benchmark.py

clean:
	rm -f mc_serial$(EXT) mc_openmp$(EXT) mc_cuda$(EXT)
