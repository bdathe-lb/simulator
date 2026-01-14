# --------------------------------------------------------
# File: algorithms/dynamic_pi.py
# --------------------------------------------------------

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from algorithms.base import BaseAlgorithm
from core.task import Task

# Constants
INFINITY = float('inf')

class DynamicPI(BaseAlgorithm):
    """
    Implements the V3 Optimized Dynamic Algorithm (Standalone).

    This class consolidates logic from Original PI, V1, and V2 into a single
    optimized implementation. It features tiered task inclusion, pinned 2-opt
    path optimization, and robust handling for dynamic environments (congestion
    awareness, blacklisting, and execution locking).
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the V3 algorithm with all necessary state variables.
        """
        super().__init__(agent_id, max_tasks, all_tasks)
        
        num_tasks = len(all_tasks)

        # --- Basic PI State (Originally from OriginalPI) ---
        self.tasks_sequence_list: List[int] = []
        self.significance_list: List[float] = [INFINITY] * num_tasks
        # Note: assigned_agent_list tracks who we think owns the task
        self.assigned_agent_list: List[int] = [-1] * num_tasks 
        self.marginal_significance_list: List[float] = [0.0] * num_tasks
        
        # --- Robust State (Originally from RobustDynamicPI) ---
        # Timestamps for Scalar Clock Consensus
        self.task_timestamps: List[float] = [0.0] * num_tasks
        # Blacklist: {task_id -> expiration_time}
        self.blacklist: Dict[int, float] = {}
        # Execution Locks: {task_id -> agent_id} (Who is physically executing what)
        self.known_execution_locks: Dict[int, int] = {}
        
        # Internal State
        self.current_time = 0.0
        self._is_currently_congested = False
        self.agent = None
        self.feasibility_matrix = []

        # --- Configuration Parameters ---
        self.BLACKLIST_DURATION = 10.0
        self.PASSIVE_COOLING_DURATION = 3.0
        self.STALE_TIMEOUT = 30.0 
        self.INERTIA_THRESHOLD = 15.0 
        self.HARD_LOCK_SIG = -1.0
        self.CONGESTION_RADIUS = 20.0 
        self.CONGESTION_PENALTY = 2000.0 

    # *******************************************************
    # Interface Implementation (From BaseAlgorithm)
    # *******************************************************

    def bind_agent(self, agent: Any) -> None:
        self.agent = agent

    def finalize_setup(self, feasibility_matrix: List[List[bool]]) -> None:
        self.feasibility_matrix = feasibility_matrix

    def get_plan(self) -> List[int]:
        return self.tasks_sequence_list

    def pack_message(self) -> Dict[str, Any]:
        """
        Packs state for broadcast. Includes Hard Lock logic from V2.
        """
        # [Fix] Ensure agent is bound before accessing attributes
        assert self.agent is not None, "Agent must be bound before packing messages"

        broadcast_sig = list(self.significance_list)
        
        # Override bid with HARD_LOCK if physically executing
        if self.agent.status == 'EXECUTING' and self.agent.current_target_task_id is not None:
            tid = self.agent.current_target_task_id
            broadcast_sig[tid] = self.HARD_LOCK_SIG

        return {
            "sender_id": self.agent_id,
            "significance_list": broadcast_sig,
            "assigned_agent_list": self.assigned_agent_list,
            "sender_target_id": self.agent.current_target_task_id,
            "task_timestamps": self.task_timestamps,
            "sender_status": self.agent.status,
            "sender_position": self.agent.position
        }

    # *******************************************************
    # Dynamic Event Hooks
    # *******************************************************

    def on_task_completed(self, task_id: int) -> None:
        if task_id in self.tasks_sequence_list:
            self.tasks_sequence_list.remove(task_id)
        if task_id in self.known_execution_locks:
            del self.known_execution_locks[task_id]

    def on_task_locked(self, task_id: int) -> None:
        self.task_timestamps[task_id] = self.current_time
        self.known_execution_locks[task_id] = self.agent_id
        self.significance_list[task_id] = self.HARD_LOCK_SIG

    # *******************************************************
    # Main Loop
    # *******************************************************

    def run_iteration(self, messages: List[Dict[str, Any]]) -> None:
        # [Fix] Ensure agent is bound for the iteration
        assert self.agent is not None, "Agent must be bound to run iteration"

        self.current_time += 1.0

        # 1. Update Congestion Status
        self._is_currently_congested = self._calculate_congestion_status(messages)

        # 2. Maintenance
        self._check_stale_information()
        
        # 3. Phases
        self._task_inclusion_phase() # V3 Tiered Logic
        self._consensus_phase(messages) # V2 Robust Logic
        self._task_removal_phase()   # V3 Optimized Removal
        
        # 4. Cleanup
        self._cleanup_blacklist()

    # *******************************************************
    # Phase 1: Inclusion (V3 Tiered Strategy)
    # *******************************************************

    def _task_inclusion_phase(self) -> None:
        """
        V3 Strategy: Tiered Sorting + Greedy Insertion.
        """
        # [Fix] Ensure agent exists
        assert self.agent is not None

        # Early exit if full and busy
        if self.agent.status == 'EXECUTING' and len(self.tasks_sequence_list) >= self.max_tasks:
            return
        
        is_desperate = (len(self.tasks_sequence_list) == 0)
        penalty = self.CONGESTION_PENALTY if (self._is_currently_congested and not is_desperate) else 0.0

        # 1. Filter Candidates
        candidate_tasks = [
            t for t in self.all_tasks 
            if not t.completed and 
            t.id not in self.blacklist and 
            t.id not in self.tasks_sequence_list and 
            self.feasibility_matrix[self.agent_id][t.id]
        ]
        
        # 2. Tiered Sorting Strategy
        URGENCY_THRESHOLD = 150.0 
        current_t = self.current_time
        
        def tiered_sort_key(task):
            # [Fix] self.agent is already asserted not None above
            time_left = task.deadline - current_t
            if time_left < URGENCY_THRESHOLD:
                # Tier 1: Critical (Sort by Deadline)
                return (0, task.deadline)
            else:
                # Tier 2: Safe (Sort by Distance)
                dist = np.linalg.norm(task.position - self.agent.position) # type: ignore
                return (1, dist)

        candidate_tasks.sort(key=tiered_sort_key)

        # 3. Greedy Insertion
        while len(self.tasks_sequence_list) < self.max_tasks:
            max_diff = -INFINITY
            best_task = -1
            best_pos = -1

            for task in candidate_tasks:
                # Skip locked tasks
                if self.significance_list[task.id] == self.HARD_LOCK_SIG: 
                    continue
                if task.id in self.known_execution_locks:
                    if self.known_execution_locks[task.id] != self.agent_id: 
                        continue

                # Calculate Marginal
                marginal, pos = self._calculate_marginal_significance(task)
                marginal += penalty

                # Strict Deadline Check
                if math.isinf(marginal): 
                    continue
                arrival_time = self.current_time + self._calculate_reach_time(task.id, pos)
                if arrival_time > task.deadline: 
                    continue

                # Calculate Gain
                diff = self.significance_list[task.id] - marginal

                if diff > 1e-4 and diff > max_diff:
                    max_diff = diff
                    best_task = task.id
                    best_pos = pos
            
            if max_diff > 0:
                self.tasks_sequence_list.insert(best_pos, best_task)
                self.significance_list[best_task] = max_diff # Store gain as significance
                self.assigned_agent_list[best_task] = self.agent_id
                self.task_timestamps[best_task] = self.current_time
                # Remove from candidates
                candidate_tasks = [t for t in candidate_tasks if t.id != best_task]
            else:
                break
        
        # 4. Post-Inclusion Optimization
        self._optimize_path_with_pinned_2opt()

        # 5. Cascade Update
        self._update_internal_significance(penalty)

    # *******************************************************
    # Phase 2: Consensus (V2 Robust Logic)
    # *******************************************************

    def _consensus_phase(self, messages: List[Dict]) -> None:
        """
        Resolves conflicts using Scalar Timestamps, Inertia, and Hard Locks.
        """
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        # Part A: Update Execution Locks Cache
        for msg in messages:
            n_id = msg['sender_id']
            n_status = msg.get('sender_status', 'IDLE')
            n_target = msg.get('sender_target_id', None)

            if n_status == 'EXECUTING' and n_target is not None:
                self.known_execution_locks[n_target] = n_id
            
            # Remove stale locks
            keys_to_delete = [tid for tid, owner in self.known_execution_locks.items() 
                              if owner == n_id and (n_status != 'EXECUTING' or n_target != tid)]
            for k in keys_to_delete:
                del self.known_execution_locks[k]

        # Part B: Bidding & Consensus
        for msg in messages:
            n_id = msg['sender_id']
            n_sig = msg['significance_list']
            n_assign = msg['assigned_agent_list']
            n_ts = msg['task_timestamps']

            num_tasks = len(self.all_tasks)
            for tid in range(num_tasks):
                # Rule 1: Self Executing -> I win
                am_executing = (self.agent.status == 'EXECUTING' and 
                                self.agent.current_target_task_id == tid)
                if am_executing:
                    self.task_timestamps[tid] = self.current_time
                    continue

                # Rule 2: Neighbor Hard Lock -> They win
                if n_sig[tid] == self.HARD_LOCK_SIG:
                     if tid in self.tasks_sequence_list or self.assigned_agent_list[tid] == self.agent_id:
                          self.blacklist[tid] = self.current_time + self.BLACKLIST_DURATION
                     
                     self.significance_list[tid] = self.HARD_LOCK_SIG
                     self.assigned_agent_list[tid] = n_id
                     self.task_timestamps[tid] = max(self.task_timestamps[tid], n_ts[tid], self.current_time)
                     continue
                
                # Rule 3: Standard Consensus (Scalar Timestamp + Inertia)
                ts_local = self.task_timestamps[tid]
                ts_neighbor = n_ts[tid]
                i_am_owner = (self.assigned_agent_list[tid] == self.agent_id)
                
                # Case A: Neighbor is strictly newer
                if ts_neighbor > ts_local:
                    if i_am_owner:
                        # Inertia Check
                        if n_sig[tid] < self.significance_list[tid] - self.INERTIA_THRESHOLD:
                            self._accept_neighbor_info(tid, n_sig, n_assign, ts_neighbor)
                        else:
                            self.task_timestamps[tid] = max(ts_local, ts_neighbor)
                    else:
                         self._accept_neighbor_info(tid, n_sig, n_assign, ts_neighbor)

                # Case B: Concurrent (Approx Equal)
                elif math.isclose(ts_neighbor, ts_local):
                    threshold = self.INERTIA_THRESHOLD if i_am_owner else 0.0
                    if n_sig[tid] < self.significance_list[tid] - threshold:
                         self._accept_neighbor_info(tid, n_sig, n_assign, ts_neighbor)
                
                # Consistency Check
                if n_id == self.assigned_agent_list[tid] and self.significance_list[tid] != self.HARD_LOCK_SIG:
                     if abs(n_sig[tid] - self.significance_list[tid]) > 1e-6:
                         self.significance_list[tid] = n_sig[tid]

    def _accept_neighbor_info(self, tid, n_sig, n_assign, ts):
        self.significance_list[tid] = n_sig[tid]
        self.assigned_agent_list[tid] = n_assign[tid]
        self.task_timestamps[tid] = ts

    # *******************************************************
    # Phase 3: Removal (V3 Optimization + V2 Logic)
    # *******************************************************

    def _task_removal_phase(self) -> None:
        """
        Combines V3's Pre-Removal Optimization with V2's Removal Logic.
        """
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        # 1. Optimization Attempt
        self._optimize_path_with_pinned_2opt()
        
        # 2. Update Significance before checking removal conditions
        penalty = self.CONGESTION_PENALTY if self._is_currently_congested else 0.0
        self._update_internal_significance(penalty)

        # 3. Standard Removal Logic (Copied from V2)
        am_executing = (self.agent.status == 'EXECUTING' and self.agent.current_target_task_id is not None)
        executing_tid = self.agent.current_target_task_id if am_executing else -1
        
        to_remove = []
        temp_sequence = self.tasks_sequence_list.copy()

        for tid in temp_sequence:
            if tid == executing_tid: 
                continue

            # Reason 1: Lost Ownership
            if self.assigned_agent_list[tid] != self.agent_id:
                to_remove.append((tid, "PASSIVE_LOST_OWNERSHIP"))
                continue
            
            # Reason 2: Constraint Violation (Deadline)
            # Note: _calculate_significance uses _calculate_total_path_cost which returns INF on violation
            real_sig = self._calculate_significance(self.all_tasks[tid])
            if math.isinf(real_sig):
                to_remove.append((tid, "ACTIVE_CONSTRAINT_VIOLATION"))
                continue

            # Reason 3: Congestion Pruning
            if self._is_currently_congested:
                to_remove.append((tid, "CONGESTION_AVOIDANCE"))

        # Execute Removal
        for tid, reason in to_remove:
            if tid in self.tasks_sequence_list:
                self.tasks_sequence_list.remove(tid)
                
                if reason == "PASSIVE_LOST_OWNERSHIP":
                     self.blacklist[tid] = self.current_time + self.PASSIVE_COOLING_DURATION
                elif reason == "CONGESTION_AVOIDANCE":
                     self.blacklist[tid] = self.current_time + 5.0

            # Reset internal state
            if not self.all_tasks[tid].completed and self.assigned_agent_list[tid] == self.agent_id:
                self.assigned_agent_list[tid] = -1
                self.significance_list[tid] = INFINITY
                self.task_timestamps[tid] = self.current_time
                self.blacklist[tid] = self.current_time + self.BLACKLIST_DURATION

    # *******************************************************
    # Optimization Logic (V3 Specific)
    # *******************************************************

    def _optimize_path_with_pinned_2opt(self) -> None:
        """
        Applies 2-Opt Local Search while pinning the active target.
        """
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        path = self.tasks_sequence_list
        n = len(path)
        if n < 3: 
            return

        # Pinning Logic
        start_index = 0
        if self.agent.status != 'IDLE' and self.agent.current_target_task_id is not None:
            if path and path[0] == self.agent.current_target_task_id:
                start_index = 1
        
        if n - start_index < 2: 
            return

        improved = True
        while improved:
            improved = False
            for i in range(start_index, n - 1):
                for j in range(i + 1, n):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    
                    current_cost = self._calculate_total_path_cost(path)
                    new_cost = self._calculate_total_path_cost(new_path)

                    if new_cost < current_cost and not math.isinf(new_cost):
                        # Safety check for feasibility
                        if self._is_path_feasible(new_path):
                            path = new_path
                            self.tasks_sequence_list = new_path
                            improved = True
                            break 
                if improved: 
                    break

    def _is_path_feasible(self, path: List[int]) -> bool:
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        curr_pos = self.agent.position
        curr_time = self.current_time
        for tid in path:
            task = self.all_tasks[tid]
            dist = np.linalg.norm(task.position - curr_pos)
            arrival = curr_time + (dist / self.agent.speed)
            if arrival > task.deadline: 
                return False
            curr_time = arrival + task.exec_duration
            curr_pos = task.position
        return True

    # *******************************************************
    # Helpers (Maintenance & Congestion)
    # *******************************************************

    def _calculate_congestion_status(self, messages: List[Dict]) -> bool:
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        my_pos = self.agent.position
        for msg in messages:
            n_id = msg['sender_id']
            n_pos = msg.get('sender_position')
            if n_pos is not None:
                dist = np.linalg.norm(my_pos - n_pos)
                # Congested if close to a higher priority (lower ID) agent
                if dist < self.CONGESTION_RADIUS and n_id < self.agent_id:
                    return True
        return False

    def _check_stale_information(self) -> None:
        for tid in range(len(self.all_tasks)):
            if self.all_tasks[tid].completed: 
                continue
            if self.assigned_agent_list[tid] == self.agent_id: 
                continue
            if self.assigned_agent_list[tid] == -1: 
                continue
            if self.significance_list[tid] == self.HARD_LOCK_SIG: 
                continue

            if (self.current_time - self.task_timestamps[tid]) > self.STALE_TIMEOUT:
                self.assigned_agent_list[tid] = -1
                self.significance_list[tid] = INFINITY
                self.task_timestamps[tid] = self.current_time
                if tid in self.known_execution_locks:
                    del self.known_execution_locks[tid]

    def _cleanup_blacklist(self) -> None:
        expired = [tid for tid, expiry in self.blacklist.items() if self.current_time >= expiry]
        for tid in expired:
            del self.blacklist[tid]
    
    def _update_internal_significance(self, penalty: float) -> None:
        """Helper to update local bids after path changes."""
        for tid in self.tasks_sequence_list:
            new_sig = self._calculate_significance(self.all_tasks[tid])
            if self._is_currently_congested: 
                new_sig += penalty
            if abs(new_sig - self.significance_list[tid]) > 1e-6:
                self.significance_list[tid] = new_sig

    # *******************************************************
    # Cost Calculations (Robust V2 Version)
    # *******************************************************

    def _calculate_total_path_cost(self, path: List[int]) -> float:
        """
        Calculates cost with STRICT Deadline constraints. 
        Returns INFINITY if any deadline is violated.
        """
        # [Fix] Use explicit None check so type checker knows it's safe afterwards
        if self.agent is None: 
            return INFINITY

        cost = 0.0
        curr_pos = self.agent.position
        current_arrival_time = self.current_time 

        for tid in path:
            task = self.all_tasks[tid]
            dist = np.linalg.norm(task.position - curr_pos)
            travel_time = dist / self.agent.speed
            
            current_arrival_time += travel_time
            if current_arrival_time > task.deadline:
                return INFINITY

            cost += travel_time + task.exec_duration
            current_arrival_time += task.exec_duration
            curr_pos = task.position

        return cost

    def _calculate_significance(self, task: Task) -> float:
        if task.id not in self.tasks_sequence_list: 
            return INFINITY
        
        cost_with = self._calculate_total_path_cost(self.tasks_sequence_list)
        
        # If path is invalid, significance is compromised (return INF to trigger removal)
        if math.isinf(cost_with):
            return INFINITY

        temp = self.tasks_sequence_list.copy()
        temp.remove(task.id)
        cost_without = self._calculate_total_path_cost(temp)
        
        return cost_with - cost_without

    def _calculate_marginal_significance(self, task: Task) -> Tuple[float, int]:
        min_marginal = INFINITY
        best_pos = -1
        base_cost = self._calculate_total_path_cost(self.tasks_sequence_list)
        
        # If base path is already broken, we can't add anything
        if math.isinf(base_cost) and len(self.tasks_sequence_list) > 0:
             return INFINITY, -1

        for i in range(len(self.tasks_sequence_list) + 1):
            temp = self.tasks_sequence_list.copy()
            temp.insert(i, task.id)
            new_cost = self._calculate_total_path_cost(temp)
            
            if math.isinf(new_cost):
                continue

            marginal = new_cost - base_cost
            if marginal < min_marginal:
                min_marginal = marginal
                best_pos = i
        return min_marginal, best_pos

    def _calculate_reach_time(self, task_id: int, pos: int) -> float:
        # [Fix] Ensure agent is bound
        assert self.agent is not None

        cost = 0.0
        curr_pos = self.agent.position
        for tid in self.tasks_sequence_list[:pos]:
            t = self.all_tasks[tid]
            cost += np.linalg.norm(t.position - curr_pos) / self.agent.speed + t.exec_duration
            curr_pos = t.position
        target = self.all_tasks[task_id]
        return cost + np.linalg.norm(target.position - curr_pos) / self.agent.speed
