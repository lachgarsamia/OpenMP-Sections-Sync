# Makefile for Exercise 1: Work Distribution with Parallel Sections
# TP4 - OpenMP (Parallel Sections, Single, Master, Synchronization)
# Author: Samia Lachgar

# Use gcc-14 on macOS for OpenMP support (brew install gcc)
CC = gcc-14
CFLAGS = -Wall -Wextra -O2 -fopenmp -std=c11
LDFLAGS = -fopenmp -lm

TARGET = sections_stats
SRC = main.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	@echo "Running with OMP_NUM_THREADS=$(OMP_NUM_THREADS)"
	./$(TARGET)

run2: $(TARGET)
	OMP_NUM_THREADS=2 ./$(TARGET)

run4: $(TARGET)
	OMP_NUM_THREADS=4 ./$(TARGET)

clean:
	rm -f $(TARGET)
