# --------------------------------------------------------
# File: algorithms/original_pi.py
# Implemented PI-Avg. based on the paper.
# --------------------------------------------------------

import numpy as np
import math
from typing import List, Dict, Any, Tuple
from algorithms.base import BaseAlgorithm
from core.agent import Agent
from core.task import Task

# Use positive infinity for initial cost values
INFINITY = float('inf')

class OriginalPI(BaseAlgorithm):
    """
    Implements the original PI Algo.

    This class provides the core logic for the Performance Impact (PI) algorithm,
    which is a distributed task allocation method based on market-based auctions.
    It iterates through three phases: Task Inclusion, Consensus, and Task Removal.

    References:
        Zhao et al., "A Heuristic Distributed Task Allocation Method for Multivehicle
        Multitask Problems and Its Application to Search and Rescue Scenario",
        [cite_start]IEEE Transactions on Cybernetics, 2016. [cite: 3]

    Features:
        - Uses Vector Clocks (`timestamp_list`) for consensus to resolve conflicts.
        - Designed primarily for static allocation phases where agents reach.
          consensus before execution.
        - Implements the iterative greedy removal strategy for handling task synergies.
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the PI algorithm state.

        Args:
            agent_id: The unique ID of the agent running this algorithm.
            max_tasks: The maximum number of tasks this agent can carry.
            all_tasks: A reference list of all available tasks in the environment.
        """
        super().__init__(agent_id, max_tasks, all_tasks)

        num_tasks = len(all_tasks)
        
        # --- PI Algorithm State ---
        # a: The current sequence of tasks assigned to the agent
        self.tasks_sequence_list: List[int] = []
        # γ: The significance list
        self.significance_list: List[float] = [INFINITY] * num_tasks
        # γ*: The marginal significance list
        self.marginal_significance_list: List[float] = [0.0] * num_tasks
        # β: The assigned list
        self.assigned_agent_list: List[int] = [-1] * num_tasks
        
        # --- Consensus State ---
        # δ: the timestamp list
        self.timestamp_list: List[float] = [] 
        # Reference to the physical agent
        self.agent: Any = None  
        # Iteration step
        self.current_time = 0.0
        
    def bind_agent(self, agent: Agent):
        """
        Binds the physical agent instance to the algorithm.

        This allows the algorithm to access the agent's current position and speed
        for cost calculations.

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
        """
        Returns the current planned sequence of tasks.
        """
        return self.tasks_sequence_list

    def pack_message(self) -> Dict[str, Any]:
        """
        Packs the internal state into a dictionary for broadcasting.

        Returns:
            A dictionary containing:
                - sender_id: ID of this agent.
                - significance_list: The list of bids.
                - assigned_agent_list: The list of task owners.
                - timestamp_list: The vector clock.
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

        Removes the completed task from the internal sequence to prevent re-execution
        and potential infinite loops during the execution phase.

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
        # --- 1. Tasks Inclusion ---
        self._task_inclusion_phase()
        
        # --- 2. Information Consensus ---
        self._consensus_phase(messages)
        
        # --- 3. Tasks Removal ---
        self._task_removal_phase()

        # Update local clock
        self.current_time += 1.0
        if self.timestamp_list:
            self.timestamp_list[self.agent_id] = self.current_time

    # ******************************* 
    # Phase Implementations
    # ******************************* 
    def _task_inclusion_phase(self) -> None:
        """
        Greedily adds tasks to the bundle.

        Iteratively finds the best unassigned task that maximizes the bid improvement
        (difference between current network bid and my marginal cost) and inserts it
        into the best position in the path.
        """
        while len(self.tasks_sequence_list) < self.max_tasks:
            max_diff = -1.0
            best_task = -1
            best_pos = -1

            for task in self.all_tasks:
                # Skip if already in my list
                if task.id in self.tasks_sequence_list: 
                    continue
                # Skip if I cannot physically perform this task type
                if not self.feasibility_matrix[self.agent_id][task.id]: 
                    continue

                # Calculate cost to insert this task
                marginal, pos = self._calculate_marginal_significance(task)
                
                # Check constraint 5: Deadline
                if self._calculate_reach_time(task.id, pos) >= task.deadline:
                    continue

                # Store the marginal significance
                self.marginal_significance_list[task.id] = marginal

                # Calculate potential gain
                # Find the task that maximizes cost reduction
                diff = self.significance_list[task.id] - marginal

                if diff > max_diff:
                    max_diff = diff
                    best_task = task.id
                    best_pos = pos

            # If an optimizable task is found, add it to the task list
            if max_diff > 0:
                self.tasks_sequence_list.insert(best_pos, best_task)
                self.significance_list[best_task] = self.marginal_significance_list[best_task]
                self.assigned_agent_list[best_task] = self.agent_id
            else:
                # No more tasks can be added profitably
                break

        # Cascade Update: Adding a task changes costs for all tasks in the path
        # We must update the significance values for all owned tasks
        for tid in self.tasks_sequence_list:
            self.significance_list[tid] = self._calculate_significance(self.all_tasks[tid])

    def _consensus_phase(self, messages: List[Dict]) -> None:
        """
        Resolves conflicts by comparing bids with neighbors.

        Uses standard consensus rules:
            1. Update vector clock to max values.
            2. Compare bids (significance). Lower bid (cost) wins.
            3. Use agent ID as tie-breaker.

        Args:
            messages: Incoming messages from neighbors.
        """
        for msg in messages:
            neighbor_sig = msg['significance_list']
            neighbor_assign = msg['assigned_agent_list']
            neighbor_time = msg['timestamp_list']

            # Update vector clock
            for i in range(len(self.timestamp_list)):
                self.timestamp_list[i] = max(self.timestamp_list[i], neighbor_time[i])
            
            # Update significance for each task
            for tid in range(len(self.all_tasks)):
                y_kj = neighbor_sig[tid]
                z_kj = neighbor_assign[tid]
                y_ij = self.significance_list[tid]
                z_ij = self.assigned_agent_list[tid]
                
                # Update Rule: Adopt neighbor's info if their significance is better (lower)
                if y_kj < y_ij:
                    self.significance_list[tid] = y_kj
                    self.assigned_agent_list[tid] = z_kj
                # Tie-breaker: If significance are equal, lower Agent ID wins
                elif math.isclose(y_kj, y_ij) and z_kj < z_ij:
                    self.significance_list[tid] = y_kj
                    self.assigned_agent_list[tid] = z_kj

    def _task_removal_phase(self) -> None:
        """
        Removes tasks that are invalid or outbid.

        Implements the Iterative Greedy Removal strategy proposed in Zhao et al. (2016).
        Instead of removing all invalid tasks at once, it iteratively removes the task
        whose removal causes the least "loss" (or most "gain"), re-evaluating synergies
        after each removal.
        """
        # --- 1. Identify the initial set of tasks that MUST be checked for removal ---
        tasks_to_check_for_removal = {
            task_id
            for task_id in self.tasks_sequence_list
            if self.assigned_agent_list[task_id] != self.agent.id
        }

        # --- 2. Iteratively remove tasks until the set is clean ---
        while tasks_to_check_for_removal:
            max_reduction = -INFINITY
            task_to_remove = -1
            # Evaluate which removal is "best"
            # Find the task whose removal can maximize improvement (reduction) in cost
            for task_id in tasks_to_check_for_removal:
                current_significance = self._calculate_significance(self.all_tasks[task_id])
                global_significance = self.significance_list[task_id]

                reduction = current_significance - global_significance
                if reduction > max_reduction:
                    max_reduction = reduction
                    task_to_remove = task_id
            
            # --- 3. Execute removal ---
            if max_reduction > 0:
                self.tasks_sequence_list.remove(task_to_remove)
                tasks_to_check_for_removal.remove(task_to_remove)
            else:
                break 

    # ********************************** 
    # Math Helpers (Cost Calculation)
    # ********************************** 
    def _calculate_total_path_cost(self, path: List[int]) -> float:
        """
        Calculates the total time cost to execute a sequence of tasks.

        Args:
            path: A list of task IDs representing the execution order.

        Returns:
            Total time (travel + execution) in seconds. Returns INFINITY if agent is unbound.
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

    def _calculate_significance(self, task: Task) -> float:
        """
        Calculates the significance (marginal cost contribution) of a task.

        Significance is defined as: Cost(Path) - Cost(Path without task).

        Args:
            task: The task object to evaluate.

        Returns:
            The marginal cost value. Returns INFINITY if task is not in the list.
        """
        if task.id not in self.tasks_sequence_list: 
            return INFINITY

        cost_with = self._calculate_total_path_cost(self.tasks_sequence_list)

        temp = self.tasks_sequence_list.copy()
        temp.remove(task.id)
        cost_without = self._calculate_total_path_cost(temp)
        return cost_with - cost_without

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
        min_marginal = INFINITY
        best_pos = -1
        base_cost = self._calculate_total_path_cost(self.tasks_sequence_list)
        
        for i in range(len(self.tasks_sequence_list) + 1):
            temp = self.tasks_sequence_list.copy()
            temp.insert(i, task.id)
            new_cost = self._calculate_total_path_cost(temp)
            marginal = new_cost - base_cost
            if marginal < min_marginal:
                min_marginal = marginal
                best_pos = i
        return min_marginal, best_pos

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
            t = self.all_tasks[tid]
            cost += np.linalg.norm(t.position - curr_pos) / self.agent.speed + t.exec_duration
            curr_pos = t.position

        # Add travel time to the target task itself
        target = self.all_tasks[task_id]
        return cost + np.linalg.norm(target.position - curr_pos) / self.agent.speed
