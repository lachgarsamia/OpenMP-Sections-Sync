# Makefile for Exercise 3: Load Balancing with Parallel Sections
# TP4 - OpenMP (Parallel Sections, Single, Master, Synchronization)
# Author: Samia Lachgar

# Use gcc-14 on macOS for OpenMP support (brew install gcc)
CC = gcc-14
CFLAGS = -Wall -Wextra -O2 -fopenmp -std=c11
LDFLAGS = -fopenmp -lm

TARGET = load_balance
SRC = main.c

RESULTS_DIR = ../../results

.PHONY: all clean run run_quiet benchmark

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run_quiet: $(TARGET)
	./$(TARGET) --quiet

# Run with 4 threads (good for seeing load balance effects)
run4: $(TARGET)
	OMP_NUM_THREADS=4 ./$(TARGET)

# Run benchmark and save results
benchmark: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	OMP_NUM_THREADS=4 ./$(TARGET) -o $(RESULTS_DIR)/ex3_times.csv

clean:
	rm -f $(TARGET)
