#!/usr/bin/env python3

import sys
import os
import csv
import time
import argparse
import numpy as np
from typing import Dict, Any, List

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.environment import Environment
from core.task import Task
import main

# Configuration
SCENARIO_DIR = os.path.join(PROJECT_ROOT, "test", "scenarios")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results")

def _is_task_inherently_impossible(task: Task, initial_agent_states: List[Dict]) -> bool:
    """Checks if a task was physically unreachable by ANY compatible agent."""
    for agent_data in initial_agent_states:
        if agent_data['type'] != task.task_type:
            continue
        dist = np.linalg.norm(task.position - agent_data['pos'])
        min_arrival_time = dist / agent_data['speed']
        if min_arrival_time <= task.deadline:
            return False
    return True

def run_single_simulation(algo_name: str, scenario_path: str, args) -> Dict[str, Any]:
    """Runs one simulation and performs post-run feasibility analysis."""

    # 1. Configuration
    args.load = True
    args.file = scenario_path
    args.algorithm = algo_name

    # Force 'row' network for Original PI as per paper specs, unless verifying robustnes
    user_network_choice = args.network
    if algo_name == "original":
        args.network = "row"
    else:
        args.network = user_network_choice

    # 2. Setup (Load Scenario)
    # Note: setup_scenario uses args.max_tasks_per_agent to init algorithms
    tasks, agents, topology = main.setup_scenario(args)

    # Snapshot for feasibility check
    initial_agent_states = [
        {'pos': np.copy(a.position), 'speed': a.speed, 'type': a.agent_type} 
        for a in agents
    ]

    # Initialize Environment
    is_fixed = (args.network != 'radius')
    env = Environment(agents, tasks, topology, args.radius, fixed_topology=is_fixed)

    # 3. Select Generator
    if algo_name == "original":
        def chained():
            yield from env.run_static_allocation(args.max_iterations)
            yield from env.run_static_execution(2.0, args.max_time)
        generator = chained()
    else:
        generator = env.run_dynamic_simulation(2.0, args.max_time)

    # 4. Execute
    start_time = time.process_time()
    final_state = None
    try:
        for state in generator:
            final_state = state
    except Exception as e:
        print(f"  [Error] {os.path.basename(scenario_path)}: {e}")
        args.network = user_network_choice
        return {"success": False}

    cpu_time = time.process_time() - start_time

    # 5. Metrics
    incomplete_tasks = [t for t in tasks if not t.completed]
    completed_count = len(tasks) - len(incomplete_tasks)

    impossible_count = 0
    if incomplete_tasks:
        for t in incomplete_tasks:
            if _is_task_inherently_impossible(t, initial_agent_states):
                impossible_count += 1

    # Success = All POSSIBLE tasks were completed
    success = (len(incomplete_tasks) == impossible_count)

    dist = sum(a.total_distance for a in agents)
    real_end_time = final_state['time'] if final_state else args.max_time
    args.network = user_network_choice # Restore

    return {
        "scenario": os.path.basename(scenario_path),
        "algorithm": algo_name,
        "max_tasks": args.max_tasks_per_agent, # Record capacity
        "success": success,
        "makespan": real_end_time,
        "total_distance": dist,
        "cpu_time": cpu_time,
        "completed": completed_count,
        "impossible": impossible_count
    }

def batch_run(args):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Example: stats_dynamic_cap1.csv or stats_original_cap3.csv
    filename_suffix = f"{args.target_algo}_cap{args.max_tasks_per_agent}"
    csv_file = os.path.join(RESULTS_DIR, f"stats_{filename_suffix}.csv")

    scenarios = sorted([f for f in os.listdir(SCENARIO_DIR) if f.endswith(".txt")])

    print(f"Running '{args.target_algo}' (Cap={args.max_tasks_per_agent}) on {len(scenarios)} scenarios...")
    print(f"Saving to: {csv_file}")

    # Write Header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "algorithm", "max_tasks", "success", "makespan", "total_distance", "cpu_time", "completed", "impossible"])

    # Loop
    for i, scen_file in enumerate(scenarios):
        full_path = os.path.join(SCENARIO_DIR, scen_file)

        # Simple progress bar
        print(f"  [{i+1}/{len(scenarios)}] {scen_file}", end="\r")

        result = run_single_simulation(args.target_algo, full_path, args)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get("scenario"),
                result.get("algorithm"),
                result.get("max_tasks"),
                result.get("success"),
                result.get("makespan"),
                result.get("total_distance"),
                result.get("cpu_time"),
                result.get("completed"),
                result.get("impossible")
            ])

    print(f"\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required algo name
    parser.add_argument("--algo", dest="target_algo", type=str, required=True, 
                        choices=["original", "dynamic"])

    # *** This is the parameter you want to vary ***
    parser.add_argument("-m", "--max-tasks-per-agent", type=int, default=3, 
                        help="Maximum tasks an agent can carry")

    # Other settings
    parser.add_argument("-r", "--radius", type=float, default=350.0)
    parser.add_argument("--max_time", type=float, default=5000.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--network", type=str, default='radius')

    # Dummy args for compatibility
    parser.add_argument("-a", "--agents", type=int, default=6) 
    parser.add_argument("-t", "--tasks", type=int, default=12)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--algorithm", type=str, default="") 

    args = parser.parse_args()
    batch_run(args)


