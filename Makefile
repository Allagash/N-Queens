# N-Queens build.
#
#   make cpu     CPU-only binary (any platform)
#   make cuda    CUDA backend    (Linux / NVIDIA, needs nvcc)
#   make metal   Metal backend   (macOS / Apple Silicon, needs full Xcode for
#                the `metal` shader compiler, not just the Command Line Tools)
#
# Every target writes ./nq. The Metal build also writes nq_metal.metallib,
# which must stay next to the binary (or be named with NQ_METALLIB=<path>).

CC       ?= cc
CFLAGS   = -O3 -Wall -Wextra -pthread
METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

.PHONY: cpu cuda metal check-cuda-syntax clean

cpu:
	$(CC) $(CFLAGS) nq.c -o nq

cuda:
	nvcc -O3 -DUSE_CUDA=1 nq.c nq_cuda.cu -Xcompiler "-pthread -Wall -Wextra" -o nq

# Kernel flags match the original nq_metal build (-O2) so timings in
# METAL_RESULTS.md stay comparable.
nq_metal.metallib: nq_metal.metal
	$(METAL) -O2 -c nq_metal.metal -o nq_metal.air
	$(METALLIB) nq_metal.air -o nq_metal.metallib

metal: nq_metal.metallib
	clang $(CFLAGS) -DUSE_METAL=1 -fobjc-arc nq.c nq_metal.m \
	      -framework Metal -framework Foundation -o nq

# Checks that nq.c still compiles with the CUDA paths enabled, without nvcc.
check-cuda-syntax:
	$(CC) -DUSE_CUDA=1 -fsyntax-only -Wall -Wextra nq.c

clean:
	rm -f nq nq_metal.air nq_metal.metallib
