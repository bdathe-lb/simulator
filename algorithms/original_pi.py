# --------------------------------------------------------
# File: algorithms/original_pi.py
# Implemented PI-Avg. based on the paper.
# --------------------------------------------------------

import numpy as np
from typing import List, Dict, Any, Tuple
from algorithms.base import BaseAlgorithm
from core.agent import Agent
from core.task import Task

# Use positive infinity for initial cost values to represent unassigned states
INFINITY = float('inf')

class OriginalPI(BaseAlgorithm):
    """
    Implements the Original Performance Impact (PI) Algorithm.

    This class provides the core logic for the Performance Impact algorithm,
    strictly following the iterative three-phase procedure described in the paper.
    It focuses on minimizing the total global cost (time) while respecting
    temporal constraints (deadlines).

    Phases:
        1. Task Inclusion: Greedily add tasks that maximize marginal gain.
        2. Consensus: Resolve conflicts using CBBA rules (Consensus-Based Bundle Algorithm).
        3. Task Removal: Iteratively remove tasks that are outbid or become inefficient.

    References:
        Zhao et al., "A Heuristic Distributed Task Allocation Method for Multivehicle
        Multitask Problems and Its Application to Search and Rescue Scenario",
        IEEE Transactions on Cybernetics, 2016.
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """Initializes the PI algorithm state.

        Args:
            agent_id (int): The unique ID of the agent running this algorithm.
            max_tasks (int): The maximum number of tasks this agent can carry (capacity).
            all_tasks (List[Task]): A reference list of all available tasks in the environment.
        """
        super().__init__(agent_id, max_tasks, all_tasks)
        num_tasks = len(all_tasks)
        
        # --------- 1. PI Algorithm State ---------
        # a: The current sequence of tasks assigned to this agent (ordered list)
        self.tasks_sequence_list: List[int] = []
        # γ: The significance list
        self.significance_list: List[float] = [INFINITY] * num_tasks
        # β: The assigned list
        self.assigned_agent_list: List[int] = [-1] * num_tasks
        
        # --------- 2. Consensus State ---------
        # δ: the timestamp list
        self.timestamp_list: List[float] = [] 

        # Reference to the physical agent
        self.agent: Any = None
        # Iteration step
        self.current_time = 0.0
        
    def bind_agent(self, agent: Agent):
        """
        Binds the physical agent instance to the algorithm.

        Args:
            agent: The physical Agent instance.
        """
        self.agent = agent
    
    def finalize_setup(self, feasibility_matrix: List[List[bool]]) -> None:
        """
        Performs second-stage initialization with global information.

        Args:
            feasibility_matrix: A boolean matrix where M[i][j] indicates if 
                                agent i can execute task j.
        """
        self.feasibility_matrix = feasibility_matrix
        num_agents = len(feasibility_matrix)
        # Initialize vector clock with zeros for all known agents
        self.timestamp_list = [0.0] * num_agents

    def get_plan(self) -> List[int]:
        """Returns the current planned sequence of task IDs."""
        return self.tasks_sequence_list

    def pack_message(self) -> Dict[str, Any]:
        """
        Packs the internal state into a dictionary for broadcasting.

        Returns:
            A dictionary containing:
                - sender_id: ID of this agent.
                - significance_list: The list of bids/costs.
                - assigned_agent_list: The list of task owners.
                - timestamp_list: The vector clock for consensus.
        """
        return {
            "sender_id": self.agent_id,
            "significance_list": self.significance_list,
            "assigned_agent_list": self.assigned_agent_list,
            "timestamp_list": self.timestamp_list
        }

    def on_task_completed(self, task_id: int) -> None:
        """
        Callback handler for when a task is physically completed.

        Args:
            task_id: The ID of the completed task.
        """
        if task_id in self.tasks_sequence_list:
            self.tasks_sequence_list.remove(task_id)

    def run_iteration(self, messages: List[Dict[str, Any]]) -> None:
        """
        Executes one complete iteration of the PI algorithm.

        The iteration consists of three phases:
            1. Inclusion: greedily add tasks that reduce local cost.
            2. Consensus: resolve conflicts with neighbors based on bids.
            3. Removal: remove tasks that are no longer valid or optimal.

        Args:
            messages: A list of messages received from neighboring agents.
        """
        # --------- Phase 1: Tasks Inclusion ---------
        self._task_inclusion_phase()
        
        # --------- Phase 2: Information Consensus ---------
        self._consensus_phase(messages)
        
        # --------- Phase 3: Tasks Removal ---------
        self._task_removal_phase()

        # Update logical clock at the end of the iteration
        self.current_time += 1.0
        if self.timestamp_list:
            self.timestamp_list[self.agent_id] = self.current_time

    # ************************************************************** 
    # Phase Implementations
    # ************************************************************** 
    def _task_inclusion_phase(self) -> None:
        """
        Phase 1: Greedily adds tasks to the bundle.
        
        The agent iteratively selects the unassigned task that 
        maximizes the difference between the current global significance
        and the agent's marginal significance 
        """
        while len(self.tasks_sequence_list) < self.max_tasks:
            max_significance_gain = 0.0
            best_task_id = -1
            best_insertion_index = -1

            # Iterate through each task to find the best candidate
            for task in self.all_tasks:
                # Skip if already in my local schedule
                if task.id in self.tasks_sequence_list: 
                    continue

                # Skip if I cannot physically perform this task type
                if not self.feasibility_matrix[self.agent_id][task.id]: 
                    continue

                # Calculate the marginal significance if we add this task
                marginal_significance, position = self._calculate_marginal_significance(task)
                
                # If the task cannot be added due to constraints, skip it
                if marginal_significance == INFINITY:
                    continue

                # Calculate potential gain: (Current Global Significance - My Marginal Significance)
                current_global_significance = self.significance_list[task.id]
                gain = current_global_significance - marginal_significance

                # Track the task that offers the highest gain
                if gain > max_significance_gain:
                    max_significance_gain = gain
                    best_task_id = task.id
                    best_insertion_index = position

            # If a profitable task is found, add it to the sequence
            if max_significance_gain > 0 and best_task_id != -1:
                self.tasks_sequence_list.insert(best_insertion_index, best_task_id)
                
                # Update internal state for the new task
                self.significance_list[best_task_id] = self._calculate_significance(best_task_id)
                self.assigned_agent_list[best_task_id] = self.agent_id
            else:
                # Stop if no task improves the objective function
                break

        # Cascade Update: Adding a task changes the position and costs of 
        # other tasks in the sequence. Update significance for all owned tasks.
        self._update_internal_significance()

    def _consensus_phase(self, messages: List[Dict]):
        """
        Phase 2: Information Consensus using CBBA Rules.

        Resolves conflicts based on the Consensus-Based Bundle Algorithm decision rules.
        It handles information staleness using vector clocks and determines winning bids
        based on minimizing significance (cost).

        Args:
            messages: A list of message dictionaries from neighbors.
        """
        for msg in messages:
            sender_id = msg['sender_id']
            neighbor_y = msg['significance_list']
            neighbor_z = msg['assigned_agent_list']
            neighbor_s = msg['timestamp_list']

            # Iterate over every task to resolve potential conflicts
            for j in range(len(self.all_tasks)):
                # --- Aliases for CBBA logic readability ---
                k = sender_id               # Sender
                i = self.agent_id           # Self

                # State values
                z_kj = neighbor_z[j]                # Who k thinks owns j
                z_ij = self.assigned_agent_list[j]  # Who i thinks owns j
                y_kj = neighbor_y[j]                # k's cost for j
                y_ij = self.significance_list[j]    # i's cost for j

                # Timestamps
                # s_km: k's timestamp about agent m
                # s_im: i's timestamp about agent m

                action = 'LEAVE' # Default

                # --- Helper: Comparison Logic ---
                def is_neighbor_better(n_cost, my_cost, n_id, my_id):
                    """Returns True if neighbor has lower cost, or ties with lower ID."""
                    if n_cost < my_cost: 
                        return True
                    if abs(n_cost - my_cost) < 1e-6 and n_id < my_id: 
                        return True # Tie-breaker
                    return False

                # --- CBBA Decision Table Implementation ---
                # Case A: Sender thinks Sender (k) owns it
                if z_kj == k:
                    # 1. I think I (i) own it -> Direct Conflict
                    if z_ij == i:
                        if is_neighbor_better(y_kj, y_ij, k, i):
                            action = 'UPDATE'
                    # 2. I think Sender (k) owns it -> Update bid if changed
                    elif z_ij == k:
                        action = 'UPDATE'
                    # 3. I think it's unassigned -> Update
                    elif z_ij == -1:
                        action = 'UPDATE'
                    # 4. I think a third party (m) owns it
                    else:
                        m = z_ij
                        s_km = neighbor_s[m]
                        s_im = self.timestamp_list[m]
                        # Update if sender has newer info about m OR sender beats m's bid
                        if s_km > s_im or is_neighbor_better(y_kj, y_ij, k, m):
                            action = 'UPDATE'

                # Case B: Sender thinks I (i) own it
                elif z_kj == i:
                    if z_ij == i:
                        action = 'LEAVE'
                    elif z_ij == k:
                        # Sender says I have it, I say Sender has it -> Sender is confused or I am stale
                        action = 'RESET'
                    elif z_ij == -1:
                        action = 'LEAVE'
                    else: # z_ij == m
                        m = z_ij
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m]:
                            action = 'RESET' # Sender knows m is out

                # Case C: Sender thinks it's Empty (-1)
                elif z_kj == -1:
                    if z_ij == i:
                        action = 'LEAVE'
                    elif z_ij == k:
                        action = 'UPDATE' # Sender released it
                    elif z_ij == -1:
                        action = 'LEAVE'
                    else: # z_ij == m
                        m = z_ij
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m]:
                            action = 'UPDATE' # Sender knows m is out

                # Case D: Sender thinks Other (m) owns it
                else: # z_kj == m
                    m = z_kj
                    if z_ij == i:
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m] and is_neighbor_better(y_kj, y_ij, m, i):
                            action = 'UPDATE'
                    elif z_ij == k:
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m]:
                            action = 'UPDATE'
                        else:
                            action = 'RESET'
                    elif z_ij == -1:
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m]:
                            action = 'UPDATE'
                    elif z_ij == m:
                        # s_km > s_im
                        if neighbor_s[m] > self.timestamp_list[m]:
                            action = 'UPDATE'
                    else: # z_ij == n 
                        n = z_ij
                        s_km = neighbor_s[m] # Sender's knowledge of m
                        s_im = self.timestamp_list[m] # My knowledge of m
                        s_kn = neighbor_s[n] # Sender's knowledge of n
                        s_in = self.timestamp_list[n] # My knowledge of n

                        # Logic from CBBA Table Col 4 Row 4
                        if s_km > s_im and s_kn > s_in:
                            action = 'UPDATE'
                        elif s_km > s_im and is_neighbor_better(y_kj, y_ij, m, n):
                            action = 'UPDATE'
                        elif s_kn > s_in and s_im > s_km:
                            action = 'RESET'
                        else:
                            action = 'LEAVE'

                # --- Execute Action ---
                if action == 'UPDATE':
                    self.significance_list[j] = y_kj
                    self.assigned_agent_list[j] = z_kj
                elif action == 'RESET':
                    self.significance_list[j] = INFINITY
                    self.assigned_agent_list[j] = -1

            for k_idx in range(len(self.timestamp_list)):
                self.timestamp_list[k_idx] = max(self.timestamp_list[k_idx], neighbor_s[k_idx])

    def _task_removal_phase(self) -> None:
        """
        Phase 3: Iterative Greedy Task Removal.

        Identifies tasks that should be removed because:
        1. The agent lost ownership during consensus (outbid).
        2. The task is no longer efficient to keep (constraint violation or synergy loss).
        """
        # 1. Identify tasks that are definitely lost (ownership changed externally)
        tasks_to_check_for_removal = {
            task_id
            for task_id in self.tasks_sequence_list
            if self.assigned_agent_list[task_id] != self.agent_id
        }

        # 2. Iteratively remove tasks until the set is clean
        while len(tasks_to_check_for_removal) > 0:
            max_reduction = -INFINITY
            task_to_remove = -1

            # Find the task whose removal maximizes the reduction in cost discrepancy
            for tid in tasks_to_check_for_removal:
                local_significance = self._calculate_significance(tid)
                global_significance = self.significance_list[tid]

                # Difference between what I pay (local) and what the market pays (global)
                # If local > global, I am inefficient and should consider removing.
                reduction = local_significance - global_significance

                if reduction > max_reduction:
                    max_reduction = reduction
                    task_to_remove = tid
            
            # Execute removal if it helps alignment with global state
            if max_reduction > 0:
                self.tasks_sequence_list.remove(task_to_remove)
                tasks_to_check_for_removal.remove(task_to_remove)
            else:
                # If no more removals improve the situation, stop.
                break 

        # 3. Re-confirm ownership of remaining tasks
        # Any task still in my list is mine, and I update its significance based on the new path.
        for tid in self.tasks_sequence_list:
            if self.assigned_agent_list[tid] != self.agent_id:
                self.assigned_agent_list[tid] = self.agent_id
                self.significance_list[tid] = self._calculate_significance(tid)

    # ************************************************************** 
    # Math Helpers (Cost Calculation)
    # ************************************************************** 
    def _calculate_total_path_cost(self, path: List[int]) -> float:
        """
        Calculates the total time cost (travel + execution) for a task sequence.

        Args:
            sequence: A list of task IDs.

        Returns:
            Total time in seconds. Returns INFINITY if deadlines are violated.
        """
        # Safety check: Ensure agent is bound
        if not self.agent:
            return INFINITY

        cost = 0.0
        curr_pos = self.agent.position
        for tid in path:
            task = self.all_tasks[tid]
            dist = np.linalg.norm(task.position - curr_pos)
            cost += dist / self.agent.speed + task.exec_duration
            curr_pos = task.position
        return cost

    def _calculate_significance(self, task_id) -> float:
        """
        Calculates the significance (marginal cost contribution) of a task.

        Significance is defined as: Cost(Path) - Cost(Path without task).

        Args:
            task: The task object to evaluate.

        Returns:
            The marginal cost value. Returns INFINITY if task is not in the list.
        """
        if task_id not in self.tasks_sequence_list: 
            return 0.0

        # 1. Calculates current total cost (baseline cost)
        base_cost = self._calculate_total_path_cost(self.tasks_sequence_list)

        # 2. Simulate the task chain after removal
        temp_sequence = self.tasks_sequence_list.copy()
        temp_sequence.remove(task_id)

        # 3. Calculate the total cost after removal
        reduced_cost = self._calculate_total_path_cost(temp_sequence)

        return base_cost - reduced_cost

    def _calculate_marginal_significance(self, task: Task) -> Tuple[float, int]:
        """
        Finds the best insertion position and cost for a new task.

        Iterates through all possible insertion points in the current path
        to find the one that minimizes the increase in total cost.

        Args:
            task: The new task to consider adding.

        Returns:
            A tuple (min_marginal_cost, best_insertion_index).
        """
        min_marginal_cost = INFINITY
        best_position = -1

        # Calculates the execution cost of the current task sequence (baseline cost)
        base_cost = self._calculate_total_path_cost(self.tasks_sequence_list)
        
        # Iterates through all possible insertion points in the current path
        for i in range(len(self.tasks_sequence_list) + 1):
            temp_task_sequence = self.tasks_sequence_list.copy()
            temp_task_sequence.insert(i, task.id)

            # Check temporal constraints
            if not self._is_path_valid(temp_task_sequence):
                continue

            # Calculate the cost after insertion
            new_cost = self._calculate_total_path_cost(temp_task_sequence)
            marginal_cost = new_cost - base_cost
            if marginal_cost < min_marginal_cost:
                min_marginal_cost = marginal_cost
                best_position = i
        return min_marginal_cost, best_position

    def _calculate_reach_time(self, task_id: int, pos: int) -> float:
        """
        Calculates the arrival time at a specific task.

        Used to check deadline constraints.

        Args:
            task_id: The ID of the target task.
            pos: The proposed insertion index in the path.

        Returns:
            The simulation time when the agent would arrive at the task.
        """
        # Safety check: Ensure agent is bound
        if not self.agent:
            return INFINITY

        cost = 0.0
        curr_pos = self.agent.position

        # Calculate time to traverse the path up to the insertion point
        for tid in self.tasks_sequence_list[:pos]:
            task = self.all_tasks[tid]
            cost += np.linalg.norm(task.position - curr_pos) / self.agent.speed + task.exec_duration
            curr_pos = task.position

        # Add travel time to the target task itself
        target = self.all_tasks[task_id]
        return cost + np.linalg.norm(target.position - curr_pos) / self.agent.speed

    def _is_path_valid(self, sequence: List[int]) -> bool:
        """
        Checks if a task sequence satisfies all temporal constraints (deadlines).

        Args:
            sequence: List of task IDs.

        Returns:
            True if feasible, False otherwise.
        """
        current_time = self.current_time
        current_position = self.agent.position
        
        for task_id in sequence:
            task = self.all_tasks[task_id]
            
            # 1. Calculate the time required to reach the task (flight time)
            dist = np.linalg.norm(task.position - current_position)
            travel_time = dist / self.agent.speed
            
            # 2. Update estimated arrival time
            arrival_time = current_time + travel_time
            
            # 3. Critical check: does the task exceed its deadline?
            #    If any single task in the chain misses its deadline, the entire chain becomes invalid
            if arrival_time > task.deadline:
                return False
            
            # 4. Update current time and position for evaluating the next task
            #    (Task execution itself also requires duration `exec_duration`)
            current_time = arrival_time + task.exec_duration
            current_position = task.position
            
        return True

    def _update_internal_significance(self):
        """Recalculates significance values for all tasks in the local bundle."""
        for tid in self.tasks_sequence_list:
            self.significance_list[tid] = self._calculate_significance(tid)
