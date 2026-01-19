#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Path Config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results")

def load_data():
    """
    Parses CSVs to build a structured dataset including Mean and Std Dev.
    
    Data Structure:
    data[algo_name][capacity] = { 
        'success_rate': float, 
        'makespan_mean': float, 
        'makespan_std': float,
        'dist_mean': float,
        'dist_std': float
    }
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return data

    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("stats_") and f.endswith(".csv")]
    
    print(f"Found {len(files)} result files. Parsing...")

    for fname in files:
        # Expected filename: stats_{algo}_cap{m}.csv
        try:
            core = fname.replace("stats_", "").replace(".csv", "")
            parts = core.split('_cap')
            if len(parts) != 2:
                print(f"Skipping malformed file: {fname}")
                continue
            
            algo_name = parts[0]
            capacity = int(parts[1])
            
            # Read CSV stats
            total = 0
            success_count = 0
            makespans = []
            distances = []
            
            with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    # [Academic Standard]
                    # Only include metrics from SUCCESSFUL runs to avoid ceiling effects
                    if row['success'] == 'True':
                        success_count += 1
                        makespans.append(float(row['makespan']))
                        distances.append(float(row['total_distance']))
            
            # Aggregate Metrics
            success_rate = (success_count / total * 100) if total > 0 else 0
            
            # Calculate Mean and Standard Deviation
            if makespans:
                ms_mean = np.mean(makespans)
                ms_std = np.std(makespans)
                di_mean = np.mean(distances)
                di_std = np.std(distances)
            else:
                ms_mean, ms_std, di_mean, di_std = 0, 0, 0, 0
            
            # Store
            data[algo_name][capacity] = {
                'success_rate': success_rate,
                'makespan_mean': ms_mean,
                'makespan_std': ms_std,
                'dist_mean': di_mean,
                'dist_std': di_std
            }
            
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    return data

def plot_trends(data):
    """
    Draws Line Charts with Error Bands (Standard Deviation).
    X-axis = Capacity, Y-axis = Metrics
    """
    if not data:
        print("No data found.")
        return

    # Prepare for plotting
    algos = sorted(data.keys())
    all_caps = set()
    for algo in algos:
        all_caps.update(data[algo].keys())
    capacities = sorted(list(all_caps)) # e.g. [1, 2, 3, 4]

    # Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Configuration for Metrics
    # (Key_Mean, Key_Std, Y-Label, Title, Legend_Loc)
    metrics_config = [
        ('success_rate', None, 'Success Rate (%)', 'Robustness Analysis', 'lower right'),
        ('makespan_mean', 'makespan_std', 'Avg. Makespan (s)', 'Time Efficiency (Lower is Better)', 'upper right'),
        ('dist_mean', 'dist_std', 'Avg. Total Distance (m)', 'Energy Efficiency (Lower is Better)', 'upper right')
    ]

    for idx, (mean_key, std_key, ylabel, title, legend_loc) in enumerate(metrics_config):
        ax = axes[idx]
        
        for algo in enumerate(algos):
            algo_name = algo[1]
            
            # Extract x (capacities) and y (means/stds)
            x_vals = []
            y_means = []
            y_stds = []
            
            for cap in capacities:
                if cap in data[algo_name]:
                    stats = data[algo_name][cap]
                    
                    # [Academic Warning]
                    # Check for survivor bias if plotting efficiency metrics
                    if std_key is not None and stats['success_rate'] < 5.0:
                        print(f"  [Warning] {algo_name} (Cap={cap}): Success rate is low ({stats['success_rate']:.1f}%). "
                              f"Average {ylabel} might be biased (Survivor Bias).")

                    x_vals.append(cap)
                    y_means.append(stats[mean_key])
                    if std_key:
                        y_stds.append(stats[std_key])
                    else:
                        y_stds.append(0)
            
            # Style Configuration
            is_baseline = (algo_name == "original")
            label_str = "Original PI" if is_baseline else "Dynamic PI (Ours)"
            color = 'tab:blue' if is_baseline else 'tab:red'
            style = '--' if is_baseline else '-'
            marker = 'x' if is_baseline else 'o'
            
            if x_vals:
                # 1. Plot Mean Line
                ax.plot(x_vals, y_means, label=label_str, 
                        marker=marker, linestyle=style, color=color, linewidth=2, markersize=8)
                
                # 2. Plot Standard Deviation Shadow (Confidence Band)
                if std_key is not None:
                    y_means_np = np.array(y_means)
                    y_stds_np = np.array(y_stds)
                    ax.fill_between(x_vals, 
                                    y_means_np - y_stds_np, 
                                    y_means_np + y_stds_np, 
                                    color=color, alpha=0.15) # Light shadow
        
        # Formatting
        ax.set_xlabel("Agent Capacity ($C_{max}$)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks(capacities)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc=legend_loc, fontsize=10, frameon=True)

    plt.suptitle("Parameter Sensitivity Analysis: Agent Capacity", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    out_file = os.path.join(RESULTS_DIR, "capacity_sensitivity_analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Analysis chart saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    print(">>> Loading simulation data...")
    data = load_data()
    print(">>> Plotting trends...")
    plot_trends(data)
