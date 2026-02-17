#!/usr/bin/env python3
"""
TP4 - Exercise 4: Plotting Script for DMVM Benchmark Results
=============================================================

This script generates visualizations for the DMVM benchmark results:
1. CPU Time vs Threads (for each version)
2. Speedup vs Threads
3. Efficiency vs Threads
4. MFLOP/s vs Threads

Author: Samia Lachgar
Date: February 2026
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'report', 'figures')

# Input file
INPUT_FILE = os.path.join(RESULTS_DIR, 'ex4_times.csv')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'serial': '#2ecc71',
    'v1_barrier': '#3498db',
    'v2_dynamic_nowait': '#e74c3c',
    'v3_static_nowait': '#9b59b6'
}
LABELS = {
    'serial': 'Serial',
    'v1_barrier': 'V1: Implicit Barrier',
    'v2_dynamic_nowait': 'V2: Dynamic + Nowait',
    'v3_static_nowait': 'V3: Static + Nowait'
}
MARKERS = {
    'serial': 'o',
    'v1_barrier': 's',
    'v2_dynamic_nowait': '^',
    'v3_static_nowait': 'D'
}


def load_data(filepath):
    """Load benchmark results from CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: Results file not found: {filepath}")
        print("Please run the benchmark first: make benchmark")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Columns: {list(df.columns)}")
    print(f"Versions: {df['version'].unique()}")
    print(f"Thread counts: {sorted(df['threads'].unique())}")
    return df


def plot_time_vs_threads(df, n_filter=None):
    """Plot CPU time vs number of threads for each version."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter by matrix size if specified
    if n_filter is not None:
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    else:
        # Use largest matrix size
        n_filter = df['n'].max()
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df[df['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['time_sec'] * 1000, 
                   marker=MARKERS[version], 
                   color=COLORS[version],
                   label=LABELS[version],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'DMVM Execution Time vs Threads{title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(sorted(df['threads'].unique()))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_speedup_vs_threads(df, n_filter=None):
    """Plot speedup vs number of threads."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if n_filter is not None:
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    else:
        n_filter = df['n'].max()
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    
    threads = sorted(df['threads'].unique())
    
    # Plot ideal speedup
    ax.plot(threads, threads, 'k--', label='Ideal (Linear)', linewidth=1.5, alpha=0.7)
    
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df[df['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['speedup'],
                   marker=MARKERS[version],
                   color=COLORS[version],
                   label=LABELS[version],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'DMVM Speedup vs Threads{title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_efficiency_vs_threads(df, n_filter=None):
    """Plot parallel efficiency vs number of threads."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if n_filter is not None:
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    else:
        n_filter = df['n'].max()
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    
    threads = sorted(df['threads'].unique())
    
    # Ideal efficiency line
    ax.axhline(y=100, color='k', linestyle='--', label='Ideal (100%)', linewidth=1.5, alpha=0.7)
    
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df[df['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['efficiency'],
                   marker=MARKERS[version],
                   color=COLORS[version],
                   label=LABELS[version],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title(f'DMVM Parallel Efficiency vs Threads{title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(threads)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mflops_vs_threads(df, n_filter=None):
    """Plot MFLOP/s vs number of threads."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if n_filter is not None:
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    else:
        n_filter = df['n'].max()
        df = df[df['n'] == n_filter]
        title_suffix = f' (N={n_filter})'
    
    for version in ['serial', 'v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df[df['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['mflops'],
                   marker=MARKERS[version],
                   color=COLORS[version],
                   label=LABELS[version],
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('MFLOP/s', fontsize=12)
    ax.set_title(f'DMVM Performance (MFLOP/s) vs Threads{title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(sorted(df['threads'].unique()))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_version_comparison(df):
    """Create a bar chart comparing versions at different thread counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_filter = df['n'].max()
    df = df[df['n'] == n_filter]
    
    threads = sorted(df['threads'].unique())
    versions = ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']
    
    # Time comparison
    ax = axes[0]
    x = np.arange(len(threads))
    width = 0.25
    
    for i, version in enumerate(versions):
        data = df[df['version'] == version].sort_values('threads')
        times = data['time_sec'].values * 1000
        ax.bar(x + i * width, times, width, label=LABELS[version], color=COLORS[version])
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Execution Time Comparison (N={n_filter})', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(threads)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Speedup comparison
    ax = axes[1]
    for i, version in enumerate(versions):
        data = df[df['version'] == version].sort_values('threads')
        speedups = data['speedup'].values
        ax.bar(x + i * width, speedups, width, label=LABELS[version], color=COLORS[version])
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'Speedup Comparison (N={n_filter})', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(threads)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_combined_metrics(df):
    """Create a 2x2 subplot with all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_filter = df['n'].max()
    df_filtered = df[df['n'] == n_filter]
    threads = sorted(df_filtered['threads'].unique())
    
    # Time vs Threads
    ax = axes[0, 0]
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df_filtered[df_filtered['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['time_sec'] * 1000,
                   marker=MARKERS[version], color=COLORS[version],
                   label=LABELS[version], linewidth=2, markersize=6)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Execution Time (N={n_filter})')
    ax.legend(fontsize=8)
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    
    # Speedup vs Threads
    ax = axes[0, 1]
    ax.plot(threads, threads, 'k--', label='Ideal', linewidth=1.5, alpha=0.7)
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df_filtered[df_filtered['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['speedup'],
                   marker=MARKERS[version], color=COLORS[version],
                   label=LABELS[version], linewidth=2, markersize=6)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Speedup (N={n_filter})')
    ax.legend(fontsize=8)
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    
    # Efficiency vs Threads
    ax = axes[1, 0]
    ax.axhline(y=100, color='k', linestyle='--', label='Ideal', linewidth=1.5, alpha=0.7)
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df_filtered[df_filtered['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['efficiency'],
                   marker=MARKERS[version], color=COLORS[version],
                   label=LABELS[version], linewidth=2, markersize=6)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'Parallel Efficiency (N={n_filter})')
    ax.legend(fontsize=8)
    ax.set_xticks(threads)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    # MFLOP/s vs Threads
    ax = axes[1, 1]
    for version in ['v1_barrier', 'v2_dynamic_nowait', 'v3_static_nowait']:
        data = df_filtered[df_filtered['version'] == version].sort_values('threads')
        if len(data) > 0:
            ax.plot(data['threads'], data['mflops'],
                   marker=MARKERS[version], color=COLORS[version],
                   label=LABELS[version], linewidth=2, markersize=6)
    ax.set_xlabel('Threads')
    ax.set_ylabel('MFLOP/s')
    ax.set_title(f'Performance (N={n_filter})')
    ax.legend(fontsize=8)
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate all plots."""
    print("="*60)
    print("TP4 Exercise 4: Generating Plots")
    print("="*60)
    
    # Create output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load data
    df = load_data(INPUT_FILE)
    
    # Get the largest matrix size for main plots
    n_max = df['n'].max()
    
    print(f"\nGenerating plots for N={n_max}...")
    
    # Generate individual plots
    fig = plot_time_vs_threads(df, n_max)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_time_vs_threads.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_time_vs_threads.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_time_vs_threads.pdf/png")
    plt.close(fig)
    
    fig = plot_speedup_vs_threads(df, n_max)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_speedup_vs_threads.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_speedup_vs_threads.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_speedup_vs_threads.pdf/png")
    plt.close(fig)
    
    fig = plot_efficiency_vs_threads(df, n_max)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_efficiency_vs_threads.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_efficiency_vs_threads.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_efficiency_vs_threads.pdf/png")
    plt.close(fig)
    
    fig = plot_mflops_vs_threads(df, n_max)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_mflops_vs_threads.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_mflops_vs_threads.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_mflops_vs_threads.pdf/png")
    plt.close(fig)
    
    fig = plot_version_comparison(df)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_version_comparison.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_version_comparison.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_version_comparison.pdf/png")
    plt.close(fig)
    
    fig = plot_combined_metrics(df)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_combined_metrics.pdf'), 
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'ex4_combined_metrics.png'), 
                bbox_inches='tight', dpi=150)
    print("  - Saved: ex4_combined_metrics.pdf/png")
    plt.close(fig)
    
    print("\n" + "="*60)
    print("All plots saved to:", FIGURES_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
