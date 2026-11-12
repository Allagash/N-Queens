CC       = clang
METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

.PHONY: all clean test

all: nq nq_metal

# Original C solver (for correctness comparison)
nq: nq.c
	$(CC) -O3 -o nq nq.c

# Metal compute shader → AIR bytecode
nq.air: nq.metal
	$(METAL) -O2 -c nq.metal -o nq.air

# AIR → Metal library
nq.metallib: nq.air
	$(METALLIB) nq.air -o nq.metallib

# Host program (links Metal + Foundation)
nq_metal: nq_metal.m nq.metallib
	$(CC) -O2 -framework Metal -framework Foundation nq_metal.m -o nq_metal

# Quick correctness check: compare both solvers for N=4..12
test: nq nq_metal
	@echo "=== Correctness check (N=4..12) ==="
	@ok=1; \
	for n in 4 5 6 7 8 9 10 11 12; do \
		c=$$(./nq $$n       | grep "solution" | grep -o '[0-9]*' | tail -1); \
		m=$$(./nq_metal $$n | grep "solution" | grep -o '[0-9]*' | tail -1); \
		if [ "$$c" = "$$m" ]; then \
			echo "N=$$n  C=$$c  Metal=$$m  OK"; \
		else \
			echo "N=$$n  C=$$c  Metal=$$m  MISMATCH"; \
			ok=0; \
		fi; \
	done; \
	if [ $$ok -eq 1 ]; then echo "All correct."; else echo "FAILURES detected."; exit 1; fi

clean:
	rm -f nq nq_metal nq.air nq.metallib
