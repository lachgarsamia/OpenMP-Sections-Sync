# TP4 - OpenMP: Parallel Sections, Single, Master, and Synchronization

**Author:** Samia Lachgar  
**Affiliation:** College of Computing, Mohammed VI Polytechnic University (UM6P)  
**Date:** February 2026

---

## Overview

This repository contains the implementation for TP4, exploring advanced OpenMP features for shared-memory parallel programming. The lab covers:

1. **Exercise 1**: Work distribution using `parallel sections`
2. **Exercise 2**: Exclusive execution with `master` vs `single` directives
3. **Exercise 3**: Load balancing strategies with parallel sections and tasks
4. **Exercise 4**: Synchronization costs and `nowait` optimization in Dense Matrix-Vector Multiplication (DMVM)

---

## Repository Structure

```
TP4_OpenMP_Samia_Lachgar/
├── README.md                          # This file
├── report/
│   ├── report.tex                     # LaTeX report
│   └── figures/                       # Generated plots
├── src/
│   ├── ex1_sections_stats/
│   │   ├── main.c                     # Parallel sections statistics
│   │   └── Makefile
│   ├── ex2_master_single_sum/
│   │   ├── main.c                     # Master vs Single comparison
│   │   └── Makefile
│   ├── ex3_sections_load_balance/
│   │   ├── main.c                     # Load balancing analysis
│   │   └── Makefile
│   └── ex4_dmvm_barriers_nowait/
│       ├── main.c                     # DMVM with barrier analysis
│       └── Makefile
├── scripts/
│   ├── run_ex4_bench.sh               # Benchmark automation script
│   └── plot_ex4.py                    # Plotting script for Exercise 4
└── results/
    ├── ex2_times.csv                  # Exercise 2 timing results
    └── ex4_times.csv                  # Exercise 4 benchmark results
```

---

## Prerequisites

### Compiler
- GCC with OpenMP support (`-fopenmp`)
- C11 standard support

### Python (for plotting)
- Python 3.6+
- Required packages: `matplotlib`, `pandas`, `numpy`

Install Python dependencies:
```bash
pip install matplotlib pandas numpy
```

---

## Build Instructions

### Build All Exercises

```bash
# Exercise 1: Parallel Sections Statistics
cd src/ex1_sections_stats
make

# Exercise 2: Master vs Single
cd src/ex2_master_single_sum
make

# Exercise 3: Load Balancing
cd src/ex3_sections_load_balance
make

# Exercise 4: DMVM Barriers
cd src/ex4_dmvm_barriers_nowait
make
```

### Clean All
```bash
cd src/ex1_sections_stats && make clean
cd src/ex2_master_single_sum && make clean
cd src/ex3_sections_load_balance && make clean
cd src/ex4_dmvm_barriers_nowait && make clean
```

---

## Running the Exercises

### Exercise 1: Parallel Sections Statistics

Computes sum, max, and standard deviation using parallel sections.

```bash
cd src/ex1_sections_stats

# Run with default threads
./sections_stats

# Run with specific thread count
OMP_NUM_THREADS=4 ./sections_stats
```

**Key Features:**
- Demonstrates data dependency handling in sections
- Section 3 (stddev) uses sum from Section 1 without recomputation
- Two implementation approaches: two-phase and flag synchronization

---

### Exercise 2: Master vs Single

Compares `#pragma omp master` and `#pragma omp single` behavior.

```bash
cd src/ex2_master_single_sum

# Run both serial and OpenMP modes
./master_single --both

# Run benchmark with large matrix
./master_single --both --benchmark

# Save results to CSV
./master_single --both --benchmark --output ../../results/ex2_times.csv
```

**Options:**
- `--serial`: Serial mode only
- `--omp`: OpenMP mode only
- `--both`: Both modes (default)
- `--benchmark`: Use large matrix (1000×1000)
- `--output FILE`: Save results to CSV

---

### Exercise 3: Load Balancing

Analyzes load balancing with heterogeneous workloads.

```bash
cd src/ex3_sections_load_balance

# Run with verbose output
./load_balance

# Run quietly
./load_balance --quiet

# Run with 4 threads (recommended for seeing balance effects)
OMP_NUM_THREADS=4 ./load_balance

# Save results
./load_balance -o ../../results/ex3_times.csv
```

**Workloads Simulated:**
- Light: ~10ms
- Moderate: ~50ms
- Heavy: ~100ms

**Approaches Compared:**
1. Naive sections (one per task)
2. Optimized sections (heavy task split into chunks)
3. Task-based (OpenMP tasks for dynamic scheduling)

---

### Exercise 4: DMVM with Barriers and Nowait

Benchmarks Dense Matrix-Vector Multiplication with different synchronization strategies.

```bash
cd src/ex4_dmvm_barriers_nowait

# Quick run
./dmvm_benchmark

# Run with specific matrix size
./dmvm_benchmark -n 2000

# Run scaling study (1, 2, 4, 8, 16 threads)
./dmvm_benchmark --scaling -o ../../results/ex4_times.csv

# Run with specific thread count
./dmvm_benchmark -t 8 -n 2000
```

**Versions Compared:**
- V1: Implicit barrier (standard `omp for`)
- V2: `schedule(dynamic)` with `nowait`
- V3: `schedule(static)` with `nowait`

**Full Benchmark (automated):**
```bash
cd scripts
chmod +x run_ex4_bench.sh
./run_ex4_bench.sh
```

---

## Generating Plots

After running Exercise 4 benchmark:

```bash
cd scripts
python3 plot_ex4.py
```

Generated plots (saved to `report/figures/`):
- `ex4_time_vs_threads.pdf`: Execution time comparison
- `ex4_speedup_vs_threads.pdf`: Speedup analysis
- `ex4_efficiency_vs_threads.pdf`: Parallel efficiency
- `ex4_mflops_vs_threads.pdf`: Performance in MFLOP/s
- `ex4_combined_metrics.pdf`: All metrics in one figure

---

## Compiling the Report

```bash
cd report
pdflatex report.tex
pdflatex report.tex  # Run twice for TOC
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OMP_NUM_THREADS` | Number of OpenMP threads | `export OMP_NUM_THREADS=4` |
| `OMP_SCHEDULE` | Default loop schedule | `export OMP_SCHEDULE="dynamic,100"` |
| `OMP_PROC_BIND` | Thread binding | `export OMP_PROC_BIND=true` |

---

## Expected Results

### Exercise 1
- All three computation methods (serial, two-phase, flag-sync) should produce identical results
- Sum, max, and stddev values will match across implementations

### Exercise 2
- OpenMP version should show speedup for the parallel reduction
- Master vs Single behavior differences highlighted in output

### Exercise 3
| Method | Expected Speedup (4 threads) |
|--------|------------------------------|
| Naive Sections | ~1.5x |
| Optimized Sections | ~2.5x |
| Task-Based | ~3.5x |

### Exercise 4
| Version | Expected Behavior |
|---------|-------------------|
| V1 (Barrier) | Good baseline, some sync overhead |
| V2 (Dynamic+nowait) | Higher scheduling overhead |
| V3 (Static+nowait) | Best performance for uniform workload |

---

## Troubleshooting

### Compilation Errors

**"omp.h not found":**
```bash
# macOS with Homebrew
brew install libomp
export CPATH="/opt/homebrew/opt/libomp/include:$CPATH"
export LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$LIBRARY_PATH"
```

**"fopenmp not recognized":**
Ensure you're using a GCC that supports OpenMP:
```bash
# macOS: Install GCC via Homebrew
brew install gcc
# Use gcc-13 instead of gcc
```

### Runtime Issues

**Only 1 thread running:**
```bash
echo $OMP_NUM_THREADS  # Check if set
export OMP_NUM_THREADS=4
```

**Inconsistent results:**
- Check for race conditions
- Verify shared/private variable declarations
- Ensure proper barrier usage

---

## License

This project is for educational purposes as part of the HPC course at UM6P under supervision of Prof.Kissami Imad
.

---

## Contact

Samia Lachgar  
College of Computing  
Mohammed VI Polytechnic University (UM6P)
