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
    """
    Checks if a task was physically unreachable by ANY compatible agent 
    from the very beginning (t=0).
    
    Args:
        task: The task to check.
        initial_agent_states: Cached list of agent properties at t=0.
        
    Returns:
        True if NO compatible agent could reach it before deadline.
    """
    for agent_data in initial_agent_states:
        # 1. Type Constraint: Only check agents that CAN perform this task
        if agent_data['type'] != task.task_type:
            continue
            
        # 2. Physical Constraint: Min Travel Time > Deadline
        dist = np.linalg.norm(task.position - agent_data['pos'])
        min_arrival_time = dist / agent_data['speed']
        
        # If at least one agent could theoretically make it, the task is POSSIBLE.
        if min_arrival_time <= task.deadline:
            return False
            
    # If loop finishes and no one could make it -> Impossible.
    return True

def run_single_simulation(algo_name: str, scenario_path: str, args) -> Dict[str, Any]:
    """Runs one simulation and performs post-run feasibility analysis."""
    
    # 1. Configuration
    args.load = True
    args.file = scenario_path
    args.algorithm = algo_name
    
    # [Topology Hack] Original PI requires ROW topology
    user_network_choice = args.network
    if algo_name == "original":
        args.network = "row"
    else:
        args.network = user_network_choice
    
    # 2. Setup
    tasks, agents, topology = main.setup_scenario(args)
    
    # [OPTIMIZATION] Snapshot initial agent states for lazy feasibility checking later.
    # We store only what's needed to calculate reachability (Physics).
    initial_agent_states = [
        {
            'pos': np.copy(a.position), 
            'speed': a.speed, 
            'type': a.agent_type
        } 
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

    # 4. Execute Simulation (Fast Forward)
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

    # 5. Metrics & Lazy Evaluation
    
    # Identify tasks that were NOT completed
    incomplete_tasks = [t for t in tasks if not t.completed]
    completed_count = len(tasks) - len(incomplete_tasks)
    
    # [LAZY CHECK] Only if there are failures, verify if they were "excusable"
    impossible_count = 0
    if incomplete_tasks:
        for t in incomplete_tasks:
            if _is_task_inherently_impossible(t, initial_agent_states):
                impossible_count += 1
    
    # Revised Success Logic:
    # Success = (We did everything that was physically possible)
    # i.e., The only incomplete tasks are the impossible ones.
    success = (len(incomplete_tasks) == impossible_count)
    
    dist = sum(a.total_distance for a in agents)
    real_end_time = final_state['time'] if final_state else args.max_time

    # Restore args
    args.network = user_network_choice

    return {
        "scenario": os.path.basename(scenario_path),
        "algorithm": algo_name,
        "success": success,
        "makespan": real_end_time,
        "total_distance": dist,
        "cpu_time": cpu_time,
        "completed": completed_count,
        "impossible": impossible_count # Useful for debugging scenario quality
    }

def batch_run(args):
    """Iterates over all scenarios and runs the specified algorithm."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    csv_file = os.path.join(RESULTS_DIR, f"stats_{args.target_algo}.csv")
    scenarios = sorted([f for f in os.listdir(SCENARIO_DIR) if f.endswith(".txt")])
    
    print(f"Running algorithm '{args.target_algo}' on {len(scenarios)} scenarios...")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "algorithm", "success", "makespan", "total_distance", "cpu_time", "completed", "impossible"])

    for i, scen_file in enumerate(scenarios):
        full_path = os.path.join(SCENARIO_DIR, scen_file)
        
        print(f"  [{i+1}/{len(scenarios)}] Processing {scen_file}...", end="\r")
        
        result = run_single_simulation(args.target_algo, full_path, args)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get("scenario"),
                result.get("algorithm"),
                result.get("success"),
                result.get("makespan"),
                result.get("total_distance"),
                result.get("cpu_time"),
                result.get("completed"),
                result.get("impossible")
            ])
            
    print(f"\nCompleted. Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", dest="target_algo", type=str, required=True, 
                        choices=["original", "v1", "v2", "v3"])
    
    parser.add_argument("-r", "--radius", type=float, default=350.0)
    parser.add_argument("--max_time", type=float, default=5000.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("-m", "--max-tasks-per-agent", type=int, default=3)
    parser.add_argument("--network", type=str, default='radius')
    
    # Placeholders
    parser.add_argument("-a", "--agents", type=int, default=6) 
    parser.add_argument("-t", "--tasks", type=int, default=12)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument( "--algorithm", type=str, default="") 

    args = parser.parse_args()
    batch_run(args)
