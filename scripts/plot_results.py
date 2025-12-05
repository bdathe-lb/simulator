#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Path Config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results")

def load_data():
    """Loads all CSV files in the results directory."""
    data = {} # {algo_name: {'makespan': [], 'dist': [], 'success_count': 0, 'total': 0}}
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return data

    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("stats_") and f.endswith(".csv")]
    
    for fname in files:
        # Extract algo name from filename 'stats_v2.csv' -> 'v2'
        algo_name = fname.replace("stats_", "").replace(".csv", "")
        
        if algo_name not in data:
            data[algo_name] = {'makespan': [], 'dist': [], 'success_count': 0, 'total': 0}
            
        with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[algo_name]['total'] += 1
                # Only count valid metrics if the run was successful
                if row['success'] == 'True':
                    data[algo_name]['success_count'] += 1
                    data[algo_name]['makespan'].append(float(row['makespan']))
                    data[algo_name]['dist'].append(float(row['total_distance']))
    
    return data

def add_labels(ax, rects, format_str="{:.1f}"):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_comparison(data):
    """Draws intuitive Bar Charts with Error Bars."""
    algos = sorted(data.keys())
    if not algos:
        print("No data found to plot. Did you run the benchmark?")
        return

    # Prepare Data for Plotting
    success_rates = []
    avg_makespan = []
    std_makespan = []
    avg_dist = []
    std_dist = []

    for a in algos:
        # Success Rate
        total = data[a]['total']
        success = data[a]['success_count']
        rate = (success / total * 100) if total > 0 else 0
        success_rates.append(rate)

        # Makespan (Mean & Std)
        m_list = data[a]['makespan']
        if m_list:
            avg_makespan.append(np.mean(m_list))
            std_makespan.append(np.std(m_list))
        else:
            avg_makespan.append(0)
            std_makespan.append(0)

        # Distance (Mean & Std)
        d_list = data[a]['dist']
        if d_list:
            avg_dist.append(np.mean(d_list))
            std_dist.append(np.std(d_list))
        else:
            avg_dist.append(0)
            std_dist.append(0)

    # Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#95a5a6', '#3498db', '#2ecc71'] # Gray, Blue, Green
    
    # --- Chart 1: Success Rate ---
    bars1 = axes[0].bar(algos, success_rates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title("Success Rate (Higher is Better)", fontsize=14)
    axes[0].set_ylabel("Percentage (%)", fontsize=12)
    axes[0].set_ylim(0, 110)
    add_labels(axes[0], bars1, "{:.1f}%")

    # --- Chart 2: Average Makespan ---
    bars2 = axes[1].bar(algos, avg_makespan, yerr=std_makespan, capsize=5, 
                        color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_title("Avg. Completion Time (Lower is Better)", fontsize=14)
    axes[1].set_ylabel("Time (s)", fontsize=12)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    add_labels(axes[1], bars2, "{:.1f}s")

    # --- Chart 3: Average Total Distance ---
    bars3 = axes[2].bar(algos, avg_dist, yerr=std_dist, capsize=5, 
                        color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_title("Avg. Total Distance (Lower is Better)", fontsize=14)
    axes[2].set_ylabel("Distance (m)", fontsize=12)
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)
    add_labels(axes[2], bars3, "{:.0f}m")

    # Final Layout
    plt.suptitle("Algorithm Performance Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "comparison_chart_bar.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Chart saved to: {output_path}")
    
    # Show
    plt.show()

if __name__ == "__main__":
    data = load_data()
    plot_comparison(data)
