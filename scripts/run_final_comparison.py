#!/usr/bin/env python3

import os
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Path Configuration ---
# Current file is in <Project_Root>/scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Paths to other scripts
SCENARIO_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "generate_scenarios.py")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_batch.py")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results")
SCENARIO_DIR = os.path.join(PROJECT_ROOT, "test", "scenarios")

# --- 2. Experiment Configurations ---
# Scales to test: (Num_Agents, Num_Tasks)
SCALES = [
    (6, 12),
    (8, 16),
    (12, 24),
    (16, 32)
]

# *** KEY CONFIGURATION: Best vs. Specified ***
ORIGINAL_CAP = 2   # Baseline Configuration
DYNAMIC_CAP = 1    # Optimized Reactive Configuration

N_SCENARIOS = 500   # Scenarios per scale
RADIUS = 250.0

def run_command(cmd_list):
    """Executes a subprocess command safely from Project Root."""
    cmd_str = " ".join(cmd_list)
    # print(f"  [Exec] {cmd_str}") # Uncomment for verbose logging
    try:
        subprocess.run(cmd_list, check=True, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Command failed: {cmd_str}")
        exit(1)

def parse_result_csv(csv_path):
    """Reads result CSV and calculates metrics."""
    if not os.path.exists(csv_path):
        print(f"  [Warning] Result file missing: {csv_path}")
        return 0, 0, 0
    
    total = 0
    success_count = 0
    makespans = []
    dists = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if str(row['success']).strip() == 'True':
                success_count += 1
                makespans.append(float(row['makespan']))
                dists.append(float(row['total_distance']))
    
    success_rate = (success_count / total * 100) if total > 0 else 0
    avg_makespan = np.mean(makespans) if makespans else 0
    avg_dist = np.mean(dists) if dists else 0
    
    return success_rate, avg_makespan, avg_dist

def main():
    # Data structure for plotting
    data = {
        'scales': [],
        'orig_sr': [], 'orig_time': [], 'orig_dist': [],
        'dyn_sr': [],  'dyn_time': [],  'dyn_dist': []
    }

    print(f"\n{'='*70}")
    print(f">>> FINAL COMPARISON EXPERIMENT")
    print(f">>> Original PI (Cap={ORIGINAL_CAP}) vs. Dynamic PI (Cap={DYNAMIC_CAP})")
    print(f"{'='*70}\n")

    for n_agents, n_tasks in SCALES:
        scale_label = f"{n_agents}-{n_tasks}"
        print(f">>> Processing Scale: {scale_label} ... ", end="", flush=True)
        data['scales'].append(scale_label)

        # 1. Generate Scenarios (Clean slate)
        if os.path.exists(SCENARIO_DIR):
            for f in os.listdir(SCENARIO_DIR):
                if f.endswith(".txt"): os.remove(os.path.join(SCENARIO_DIR, f))
        
        run_command(["python3", SCENARIO_SCRIPT, 
                     "--agents", str(n_agents), "--tasks", str(n_tasks), "--count", str(N_SCENARIOS)])

        # 2. Run Original PI (M=2)
        csv_orig = os.path.join(RESULTS_DIR, f"stats_original_cap{ORIGINAL_CAP}.csv")
        if os.path.exists(csv_orig): os.remove(csv_orig)
        
        run_command(["python3", RUN_SCRIPT, 
                     "--algo", "original", "--network", "row", 
                     "--max-tasks-per-agent", str(ORIGINAL_CAP),
                     "--agents", str(n_agents), "--tasks", str(n_tasks)])
        
        s_orig, t_orig, d_orig = parse_result_csv(csv_orig)
        data['orig_sr'].append(s_orig)
        data['orig_time'].append(t_orig)
        data['orig_dist'].append(d_orig)

        # 3. Run Dynamic PI (M=1)
        csv_dyn = os.path.join(RESULTS_DIR, f"stats_dynamic_cap{DYNAMIC_CAP}.csv")
        if os.path.exists(csv_dyn): os.remove(csv_dyn)
        
        run_command(["python3", RUN_SCRIPT, 
                     "--algo", "dynamic", "--network", "radius", "--radius", str(RADIUS),
                     "--max-tasks-per-agent", str(DYNAMIC_CAP),
                     "--agents", str(n_agents), "--tasks", str(n_tasks)])
        
        s_dyn, t_dyn, d_dyn = parse_result_csv(csv_dyn)
        data['dyn_sr'].append(s_dyn)
        data['dyn_time'].append(t_dyn)
        data['dyn_dist'].append(d_dyn)

        print("Done.")
        print(f"    Original (M={ORIGINAL_CAP}): SR={s_orig:.1f}%, Time={t_orig:.1f}s")
        print(f"    Dynamic  (M={DYNAMIC_CAP}): SR={s_dyn:.1f}%, Time={t_dyn:.1f}s")

    # 4. Plotting
    print("\n>>> Generating Final Charts...")
    plot_final_comparison(data)

def plot_final_comparison(data):
    scales = data['scales']
    x = np.arange(len(scales))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Labels for legend
    label_orig = f'Original PI (M={ORIGINAL_CAP})'
    label_dyn = f'Dynamic PI (M={DYNAMIC_CAP})'

    def add_labels(ax, rects, fmt="{:.1f}"):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Chart 1: Success Rate ---
    ax = axes[0]
    rects1 = ax.bar(x - width/2, data['orig_sr'], width, label=label_orig, color='#4e79a7', alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, data['dyn_sr'], width, label=label_dyn, color='#e15759', alpha=0.9, edgecolor='black')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Robustness Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim(0, 115)
    ax.legend()
    add_labels(ax, rects1)
    add_labels(ax, rects2)

    # --- Chart 2: Makespan ---
    ax = axes[1]
    rects1 = ax.bar(x - width/2, data['orig_time'], width, label=label_orig, color='#4e79a7', alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, data['dyn_time'], width, label=label_dyn, color='#e15759', alpha=0.9, edgecolor='black')
    ax.set_ylabel('Avg Makespan (s)')
    ax.set_title('Efficiency: Completion Time')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    add_labels(ax, rects1)
    add_labels(ax, rects2)

    # --- Chart 3: Distance ---
    ax = axes[2]
    rects1 = ax.bar(x - width/2, data['orig_dist'], width, label=label_orig, color='#4e79a7', alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, data['dyn_dist'], width, label=label_dyn, color='#e15759', alpha=0.9, edgecolor='black')
    ax.set_ylabel('Avg Total Distance (m)')
    ax.set_title('Efficiency: Total Distance')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    add_labels(ax, rects1, "{:.0f}")
    add_labels(ax, rects2, "{:.0f}")

    plt.suptitle("Final Algorithm Comparison (Optimized Configurations)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    out_file = os.path.join(RESULTS_DIR, "final_comparison_chart.png")
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"Chart saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
