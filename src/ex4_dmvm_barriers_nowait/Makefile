# Makefile for Exercise 4: DMVM with Barriers and Nowait
# TP4 - OpenMP (Parallel Sections, Single, Master, Synchronization)
# Author: Samia Lachgar

# Use gcc-14 on macOS for OpenMP support (brew install gcc)
CC = gcc-14
CFLAGS = -Wall -Wextra -O2 -fopenmp -std=c11
LDFLAGS = -fopenmp -lm

TARGET = dmvm_benchmark
SRC = main.c

RESULTS_DIR = ../../results
SCRIPTS_DIR = ../../scripts

.PHONY: all clean run benchmark scaling

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Quick run with default settings
run: $(TARGET)
	./$(TARGET)

# Run with specific thread count
run2: $(TARGET)
	OMP_NUM_THREADS=2 ./$(TARGET)

run4: $(TARGET)
	OMP_NUM_THREADS=4 ./$(TARGET)

run8: $(TARGET)
	OMP_NUM_THREADS=8 ./$(TARGET)

# Run benchmark and save to CSV
benchmark: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) -n 2000 --scaling -o $(RESULTS_DIR)/ex4_times.csv

# Run scaling study with larger matrix
scaling_large: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) -n 4000 --scaling -o $(RESULTS_DIR)/ex4_times_large.csv

# Generate plots (requires Python with matplotlib)
plots: benchmark
	@mkdir -p ../../report/figures
	python3 $(SCRIPTS_DIR)/plot_ex4.py

clean:
	rm -f $(TARGET)
