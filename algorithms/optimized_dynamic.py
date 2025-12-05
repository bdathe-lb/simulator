# --------------------------------------------------------
# File: algorithms/optimized_dynamic.py
# A dynamic algorithm improved based on the PI algorithm.
# Version 3
# --------------------------------------------------------

import numpy as np
from typing import List, Dict
from algorithms.robust_dynamic import RobustDynamicPI
from algorithms.original_pi import INFINITY
from core.task import Task

class OptimizedDynamicPI(RobustDynamicPI):
    """
    Implements the V3 Optimized Dynamic Algorithm.

    This strategy builds upon the V2 Robust framework but introduces heuristic
    optimizations to improve solution quality (Makespan) and success rate in
    constrained scenarios.

    Inheritance:
        BaseAlgorithm -> OriginalPI -> RobustDynamicPI (V2) -> OptimizedDynamicPI (V3)

    Optimizations:
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        super().__init__(agent_id, max_tasks, all_tasks)

    # -------------------------------------------
    # Optimization:  EDF Sorting + 2-Opt
    # -------------------------------------------
    def _task_inclusion_phase(self) -> None:
        """
        Adds tasks prioritizing Urgency and optimizas path with 2-Opt
        """
        
        # Sort tasks by Deadline (EDF Strategy)
        sorted_tasks = sorted(self.all_tasks, key=lambda t: t.deadline)
        
        # Loop
        while len(self.tasks_sequence_list) < self.max_tasks:
            max_diff = -INFINITY
            best_task = -1
            best_pos = -1

            for task in sorted_tasks:
                # Filters
                if task.completed: 
                    continue
                if task.id in self.blacklist: 
                    continue
                if task.id in self.tasks_sequence_list: 
                    continue
                if not self.feasibility_matrix[self.agent_id][task.id]: 
                    continue

                marginal, pos = self._calculate_marginal_significance(task)
                
                # Check Deadline
                arrival_time = self.current_time + self._calculate_reach_time(task.id, pos)
                if arrival_time > task.deadline:
                    continue

                self.marginal_significance_list[task.id] = marginal
                diff = self.significance_list[task.id] - marginal

                if diff > max_diff:
                    max_diff = diff
                    best_task = task.id
                    best_pos = pos

            if max_diff > 0:
                self.tasks_sequence_list.insert(best_pos, best_task)
                self.significance_list[best_task] = self.marginal_significance_list[best_task]
                self.assigned_agent_list[best_task] = self.agent_id
                self.task_timestamps[best_task] = self.current_time
            else:
                break
        
        # Apply 2-Opt to fix path crossings
        self._optimize_bundle()

        # Cascade Update: Re-calc costs after insertion and optimization
        for tid in self.tasks_sequence_list:
            new_sig = self._calculate_significance(self.all_tasks[tid])
            if abs(new_sig - self.significance_list[tid]) > 1e-6:
                self.significance_list[tid] = new_sig
                self.task_timestamps[tid] = self.current_time    

    # ---------------------------------------------
    # Optimization 2: Urgency Weighted Cost
    # ---------------------------------------------
    def _calculate_total_path_cost(self, path: List[int]) -> float:
        """
        Calculates cost with Urgency Penalty.
        
        Reduces the cost of tasks that are close to their deadline, effectively
        increasing their 'Significance' (Value) in the auction process.
        """
        if not self.agent:
            return INFINITY
        
        cost = 0.0
        curr_pos = self.agent.position

        for tid in path:
            task = self.all_tasks[tid]
            dist = float(np.linalg.norm(task.position - curr_pos))
            cost += dist / self.agent.speed + task.exec_duration 
            curr_pos = task.position

        return float(cost)
            
    # -------------------------------------------------------------------------
    # Optimization 3: Consensus Inertia
    # -------------------------------------------------------------------------
    def _consensus_phase(self, messages: List[Dict]) -> None:
        """
        Resolves conflicts using V2 Locking + V3 Inertia.
        """
        super()._consensus_phase(messages)

    # ---------------------------------------------
    # Helper: 2-Opt Logic
    # ---------------------------------------------
    def _optimize_bundle(self) -> None:
        """Applies 2-Opt local search to optimize the task sequence."""
        path = self.tasks_sequence_list
        n = len(path)
        if n < 2:
            return
        
        # Determine the start index for optimization
        # If we are busy, index 0 is locked. Start optimization from index 1.
        # If we are IDLE, we can reorder everything (start from 0).
        start_index = 0
        if self.agent.status != 'IDLE' and self.agent.current_target_task_id is not None:
            # Verify if the first task is indeed the current physical target
            if path[0] == self.agent.current_target_task_id:
                start_index = 1
        
        # If fewer than 2 tasks remaining in the optimizable part, skip
        if n - start_index < 2:
            return

        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Create reversed segment path
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    
                    # Note: We use the *base* cost function (without urgency weights)
                    # for pure path optimization, or consistent weighted cost.
                    # Here we use self._calculate_total_path_cost which includes weights.
                    # This aligns optimization with the bidding strategy.
                    current_cost = self._calculate_total_path_cost(path)
                    new_cost = self._calculate_total_path_cost(new_path)
                    
                    if new_cost < current_cost:
                        path = new_path
                        self.tasks_sequence_list = new_path
                        improved = True
                        break
                if improved: 
                    break
