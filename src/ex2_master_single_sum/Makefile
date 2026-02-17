# Makefile for Exercise 2: Master vs Single Execution
# TP4 - OpenMP (Parallel Sections, Single, Master, Synchronization)
# Author: Samia Lachgar

# Use gcc-14 on macOS for OpenMP support (brew install gcc)
CC = gcc-14
CFLAGS = -Wall -Wextra -O2 -fopenmp -std=c11
LDFLAGS = -fopenmp -lm

TARGET = master_single
SRC = main.c

RESULTS_DIR = ../../results

.PHONY: all clean run run_serial run_omp benchmark

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run both modes with verbose output
run: $(TARGET)
	./$(TARGET) --both

# Run serial mode only
run_serial: $(TARGET)
	./$(TARGET) --serial

# Run OpenMP mode only
run_omp: $(TARGET)
	./$(TARGET) --omp

# Run benchmark and save results
benchmark: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) --both --benchmark --output $(RESULTS_DIR)/ex2_times.csv

# Run with different thread counts
bench_threads: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	@echo "mode,threads,time_sec" > $(RESULTS_DIR)/ex2_thread_scaling.csv
	@for t in 1 2 4 8; do \
		echo "Running with $$t threads..."; \
		OMP_NUM_THREADS=$$t ./$(TARGET) --benchmark 2>/dev/null | \
		grep "OpenMP time" | sed "s/.*: /omp,$$t,/; s/ seconds//"; \
	done >> $(RESULTS_DIR)/ex2_thread_scaling.csv

clean:
	rm -f $(TARGET)
