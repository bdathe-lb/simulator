#!/usr/bin/env python3
# scripts/run_sensitivity.py

import os
import csv
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# ... (Path config same as before) ...
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_batch.py")
SCENARIO_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "generate_scenarios.py")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results_sensitivity")

# 固定用一个中等难度的场景来测试参数
N_AGENTS = 10
N_TASKS = 20
N_SCENARIOS = 500
CAPACITIES = [1, 2, 3, 4] # 自变量

def main():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    
    # 1. Generate Scenarios (One set is enough)
    print(">>> Generating Scenarios...")
    subprocess.run(["python3", SCENARIO_SCRIPT, "--agents", str(N_AGENTS), "--tasks", str(N_TASKS), "--count", str(N_SCENARIOS)], check=True)

    success_rates = []

    # 2. Run Dynamic PI with different capacities
    for m in CAPACITIES:
        print(f">>> Testing Dynamic PI with Capacity = {m} ...")
        # 修改 run_batch.py 的输出路径逻辑可能比较麻烦，
        # 这里建议直接运行，然后去读取默认结果目录下的文件
        subprocess.run([
            "python3", RUN_SCRIPT, 
            "--algo", "dynamic", 
            "--network", "radius", 
            "--radius", "250",
            "--max-tasks-per-agent", str(m),
            "--agents", str(N_AGENTS), 
            "--tasks", str(N_TASKS)
        ], check=True)
        
        # 读取结果 (假设 run_batch 生成在 test/results/)
        # 注意：你需要确保 run_batch.py 生成的文件名包含 cap 信息
        csv_path = os.path.join(PROJECT_ROOT, "test", "results", f"stats_dynamic_cap{m}.csv")
        rate = parse_success_rate(csv_path)
        success_rates.append(rate)
        print(f"    -> Success Rate: {rate:.1f}%")

    # 3. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(CAPACITIES, success_rates, marker='o', linewidth=2, color='#e15759')
    plt.title(f'Parameter Sensitivity: Agent Capacity (Dynamic PI)', fontsize=14)
    plt.xlabel('Max Tasks Per Agent (M)')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, linestyle='--')
    plt.xticks(CAPACITIES)
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_curve.png"), dpi=200)
    print("Sensitivity chart saved.")

def parse_success_rate(csv_path):
    # ... (Standard CSV parsing logic) ...
    if not os.path.exists(csv_path): return 0
    total = 0
    success = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if str(row['success']).strip() == 'True': success += 1
    return (success/total*100) if total else 0

if __name__ == "__main__":
    main()
