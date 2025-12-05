# --------------------------------------------------------
# File: algorithms/robust_dynamic.py
# A dynamic algorithm improved based on the PI algorithm.
# Version 2
# --------------------------------------------------------

import math
from typing import List, Dict, Any
from algorithms.original_pi import OriginalPI, INFINITY
from core.task import Task

class RobustDynamicPI(OriginalPI):
    """
    Implements the V2 Robust Dynamic Algorithm.

    This strategy extends the Original PI algorithm to function reliably in
    highly dynamic environments with communication dropouts and real-time execution.

    Key Features:
        1. Scalar Task Timestamps: Replaces (or augments) vector clocks to track
           the "freshness" of information for each individual task.
        2. Blacklist (Tombstone) Mechanism: Prevents oscillation (infinite add/remove loops)
           by enforcing a cooling-off period after voluntarily abandoning a task.
        3. Execution Locking: Uses physical agent status ('EXECUTING') as a
           highest-priority signal to resolve conflicts when agents are close to a target.

    Attributes:
        task_timestamps (List[float]): Stores the simulation time of the last
            valid update for each task.
        blacklist (Dict[int, float]): Maps task IDs to their expiry timestamp.
            Tasks in this list are temporarily banned from inclusion.
        BLACKLIST_DURATION (float): Duration in seconds for the cooling-off period.
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the Robust Dynamic PI algorithm.

        Args:
            agent_id: The unique ID of the agent.
            max_tasks: The maximum number of tasks the agent can carry.
            all_tasks: A reference list of all available tasks.
        """
        super().__init__(agent_id, max_tasks, all_tasks)

        num_tasks = len(all_tasks)

        # --- V2 Specific Data Structures ---
        # Record the timestamp when the current task starts execution, for use in conflict arbitration
        self.start_execution_time = float('inf')
        # Timestamp for the information about each task
        self.task_timestamps: List[float] = [0.0] * num_tasks
        # Tombstones: {task_id, expiry_time}
        self.blacklist: Dict[int, float] = {}
        # Configuration: How long to ignore a task after dropping it (prevent oscillation)
        self.BLACKLIST_DURATION = 10.0

    def finalize_setup(self, feasibility_matrix: List[List[bool]]) -> None:
        """
        Performs second-stage initialization.

        Note: Unlike OriginalPI, V2 relies primarily on `task_timestamps`, 
        so we do not strictly need to initialize the vector clock `timestamp_list` here,
        but we keep the feasibility matrix setup.

        Args:
            feasibility_matrix: A boolean matrix for task eligibility.
        """
        self.feasibility_matrix = feasibility_matrix

    def pack_message(self) -> Dict[str, Any]:
        """
        Packs the internal state into a message dictionary for broadcasting.

        V2 adds `task_timestamps` and physical status (`sender_status`) to the message
        to enable the Execution Locking mechanism.

        Returns:
            A dictionary containing algorithm state and physical status.
        """
        return {
            "sender_id": self.agent_id,
            "significance_list": self.significance_list,
            "assigned_agent_list": self.assigned_agent_list,
            "sender_target_id": self.agent.current_target_task_id,
            # Send scalar timestamps for conflict resolution
            "task_timestamps": self.task_timestamps,
            # Send physical status for Execution Locking
            "sender_status": self.agent.status
        }
    def on_task_completed(self, task_id: int) -> None:
        """
        Callback when a task is physically completed.

        Removes the task from the execution plan but MAINTAINS ownership in
        `assigned_agent_list` so neighbors know it's done by this agent.

        Args:
            task_id: The ID of the completed task.
        """
        if task_id in self.tasks_sequence_list:
            self.tasks_sequence_list.remove(task_id)

    def on_task_locked(self, task_id: int) -> None:
        """
        Callback when the agent starts physically executing a task.

        Updates the timestamp to prioritize this "lock" state in the network.

        Args:
            task_id: The ID of the task being locked.
        """
        self.task_timestamps[task_id] = self.current_time

    def run_iteration(self, messages: List[Dict[str, Any]]) -> None:
        """
        Executes one complete iteration of the V2 algorithm.

        Updates the logical clock, performs the three PI phases, and cleans up
        expired blacklist entries.

        Args:
            messages: List of messages from neighbors.
        """
        # --- 0. Update Logical Clock ---
        self.current_time += 1.0

        # --- 1. Task Inclusion ---
        self._task_inclusion_phase()

        # --- 2. Consensus (with Time & Lock checks) ---
        self._consensus_phase(messages)

        # --- 3. Task Removal (with Tombstones) ---
        self._task_removal_phase()

        # --- 4. Cleanup Blacklist ---
        self._cleanup_blacklist()
    
    def _cleanup_blacklist(self) -> None:
        """
        Removes expired tasks from the blacklist.
        """
        expired_tasks = [tid for tid, expiry in self.blacklist.items() 
                         if self.current_time >= expiry]
        for tid in expired_tasks:
            del self.blacklist[tid]

    def _task_inclusion_phase(self) -> None:
        """
        Greedily adds tasks, respecting Blacklist and Timestamps.
        """
        # Loop
        while len(self.tasks_sequence_list) < self.max_tasks:
            max_diff = -INFINITY
            best_task = -1
            best_pos = -1

            for task in self.all_tasks:
                # If the task is already completed, skip it
                if task.completed:
                    continue
                # If task is in cooling-off period (prevent oscillation), skip it
                if task.id in self.blacklist:
                    continue
                # If the task is already in the execution list, skip it
                if task.id in self.tasks_sequence_list: 
                    continue
                # If the task does not meet the constraint conditions (feasibility matrix), skip it
                if not self.feasibility_matrix[self.agent_id][task.id]: 
                    continue

                # Calculate the marginal significance of the task
                marginal, pos = self._calculate_marginal_significance(task)
                
                # Check Deadline: Current Time + Travel + Exec <= Deadline
                arrival_time = self.current_time + self._calculate_reach_time(task.id, pos)
                if arrival_time > task.deadline:
                    continue

                self.marginal_significance_list[task.id] = marginal
                
                # Calculate Gain: Global Best - My Marginal
                diff = self.significance_list[task.id] - marginal

                if diff > max_diff:
                    max_diff = diff
                    best_task = task.id
                    best_pos = pos

            # --- Execution ---
            if max_diff > 0:
                self.tasks_sequence_list.insert(best_pos, best_task)
                self.significance_list[best_task] = self.marginal_significance_list[best_task]
                self.assigned_agent_list[best_task] = self.agent_id
                # Update timestamp because we changed the state
                self.task_timestamps[best_task] = self.current_time
            else:
                break

        # Cascade Update: Re-evaluate costs for all tasks in path
        for tid in self.tasks_sequence_list:
            new_sig = self._calculate_significance(self.all_tasks[tid])
            # If cost changed significantly, update sig and timestamp
            if abs(new_sig - self.significance_list[tid]) > 1e-6:
                self.significance_list[tid] = new_sig
                self.task_timestamps[tid] = self.current_time

    def _consensus_phase(self, messages: List[Dict]) -> None:
        """
        Resolves conflicts using Execution Locks, Timestamps, and Significance.

        Priority Order:
            1. Execution Lock (Physical State)
            2. Newer Information (Timestamp)
            3. Better Bid (Significance)
        """
        for msg in messages:
            n_id = msg['sender_id']
            n_sig = msg['significance_list']
            n_assign = msg['assigned_agent_list']
            n_ts = msg['task_timestamps']
            n_status = msg.get('sender_status', 'IDLE')
            n_target = msg.get('sender_target_id', None)

            for tid in range(len(self.all_tasks)):
                # --- Rule 0: Execution Lock (Strongest) ---
                # If neighbor is physically executing this task, they win immediately.
                if n_status == 'EXECUTING' and n_target == tid:
                    # Unless I am ALSO executing it (should be rare collision)
                    am_executing = (self.agent.status == 'EXECUTING' and 
                                    self.agent.current_target_task_id == tid)
                    if not am_executing:
                        self.significance_list[tid] = n_sig[tid]
                        self.assigned_agent_list[tid] = n_id
                        # Force update timestamp to propagate this lock
                        self.task_timestamps[tid] = max(self.task_timestamps[tid], n_ts[tid], self.current_time)
                        continue

                # --- Rule 1: Timeliness Priority ---
                ts_local = self.task_timestamps[tid]
                ts_neighbor = n_ts[tid]

                if ts_neighbor > ts_local:
                    # Neighbor has newer info (even if it's a reset/drop), accept it.
                    self.significance_list[tid] = n_sig[tid]
                    self.assigned_agent_list[tid] = n_assign[tid]
                    self.task_timestamps[tid] = ts_neighbor

                # --- Rule 2: Significance (Standard PI) ---
                elif math.isclose(ts_neighbor, ts_local):
                    # Compare costs (lower is better)
                    if n_sig[tid] < self.significance_list[tid]:
                        self.significance_list[tid] = n_sig[tid]
                        self.assigned_agent_list[tid] = n_assign[tid]
                    # Tie-breaker: Lower Agent ID wins
                    elif math.isclose(n_sig[tid], self.significance_list[tid]):
                        if n_assign[tid] != -1 and n_assign[tid] < self.assigned_agent_list[tid]:
                            self.assigned_agent_list[tid] = n_assign[tid]

    def _task_removal_phase(self) -> None:
        """
        Removes tasks and applies Tombstones/Blacklist logic.
        """
        to_remove = []
        temp_sequence = self.tasks_sequence_list.copy()
        
        #  --- 1. Identify candidates ---
        for tid in temp_sequence:
            # Passive Removal: I lost ownership in consensus
            if self.assigned_agent_list[tid] != self.agent_id:
                to_remove.append(tid)
                continue
            
            # Active Removal: Cost became infinite (constraint violation)
            real_sig = self._calculate_significance(self.all_tasks[tid])
            if math.isinf(real_sig):
                to_remove.append(tid)

        #  --- 2. Execute Removal & Apply Tombstones ---
        for tid in to_remove:
            if tid in self.tasks_sequence_list:
                self.tasks_sequence_list.remove(tid)
            
            # If I was the owner and I am dropping it (and it's not done),
            # I must explicitly reset the state to "Unassigned" and timestamp it.
            if not self.all_tasks[tid].completed and self.assigned_agent_list[tid] == self.agent_id:
                # Reset ownership
                self.assigned_agent_list[tid] = -1
                self.significance_list[tid] = INFINITY
                
                # Update timestamp so this "Reset" propagates as new info
                self.task_timestamps[tid] = self.current_time
                
                # Add to Blacklist: Don't try to pick this up again immediately
                self.blacklist[tid] = self.current_time + self.BLACKLIST_DURATION
