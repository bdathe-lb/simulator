# --------------------------------------------------------
# File: algorithms/basic_dynamic.py
# A dynamic algorithm improved based on the PI algorithm.
# Version 1
# --------------------------------------------------------

# The feature implementation is correct fine.

from typing import List
from algorithms.original_pi import INFINITY, OriginalPI
from core.task import Task

class BasicDynamicPI(OriginalPI):
    """
    Implements the V1 Basic Dynamic Algorithm.

    This algorithm extends the Original PI (CBBA) to work in a continuous
    dynamic environment. It uses the same Vector Clock consensus mechanism
    but adds hooks to handle real-time task completion and execution locking.

    Key Differences from OriginalPI:
    1. Filters out completed tasks during the inclusion phase.
    2. Updates task significance to 0.0 upon completion to broadcast success.
    3. Applies a 'soft lock' (low significance) when execution starts.
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the Basic Dynamic PI algorithm.

        Args:
            agent_id: The unique ID of the agent.
            max_tasks: The maximum number of tasks the agent can carry.
            all_tasks: A reference list of all available tasks.
        """
        super().__init__(agent_id, max_tasks, all_tasks)

    def on_task_completed(self, task_id: int) -> None:
        """
        Handles logic when a task is physically completed.

        For V1, we remove the task from the local plan and set its cost (significance)
        to 0.0. This '0 cost' will propagate through the network via consensus,
        ensuring other agents drop the task because they cannot beat a 0 cost.

        Args:
            task_id: The ID of the completed task.
        """
        # --- 1. Remove from local execution plan ---
        if task_id in self.tasks_sequence_list:
            self.tasks_sequence_list.remove(task_id)
        
        # --- 2. Update state to reflect completion ---
        self.significance_list[task_id] = 0.0
        self.assigned_agent_list[task_id] = self.agent_id

        # --- 3. Update local timestamp to ensure this new info propagates ---
        # (Since we inherited OriginalPI, we rely on vector clocks, but updating
        # local state is enough for the next consensus step to pick it up)
        pass        

    def on_task_locked(self, task_id: int) -> None:
        """
        Applies a soft lock when the agent starts executing a task.

        Sets the significance to a very low value (epsilon) to discourage
        others from snatching it during the final approach.

        Args:
            task_id: The ID of the task being executed.
        """
        # 0.0001 represents a very low cost, effectively locking the task
        # unless it is completed (0.0)
        self.significance_list[task_id] = 0.0001
        self.assigned_agent_list[task_id] = self.agent_id

    def _task_inclusion_phase(self) -> None:
        """
        Overrides inclusion phase to filter out physically completed tasks.
        
        In a dynamic simulation, 'completed' tasks persist in the environment.
        We must explicitly check `task.completed` to prevent re-adding them.
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
                # If the task is already in the execution list, skip it
                if task.id in self.tasks_sequence_list: 
                    continue
                # If the task does not meet the constraint conditions (feasibility matrix), skip it
                if not self.feasibility_matrix[self.agent_id][task.id]: 
                    continue

                # Calculate the marginal significance of the task
                marginal, pos = self._calculate_marginal_significance(task)
                
                # Correct Deadline Check: Current Time + Travel Duration <= Deadline
                arrival_time = self.current_time + self._calculate_reach_time(task.id, pos)
                if arrival_time > task.deadline:
                    continue

                # Update marginal significance list
                self.marginal_significance_list[task.id] = marginal
                
                # Compare with Global Bid
                diff = self.significance_list[task.id] - marginal

                if diff > max_diff:
                    max_diff = diff
                    best_task = task.id
                    best_pos = pos

            if max_diff > 0:
                # Found a valid task
                self.tasks_sequence_list.insert(best_pos, best_task)
                self.significance_list[best_task] = self.marginal_significance_list[best_task]
                self.assigned_agent_list[best_task] = self.agent_id
            else:
                # No beneficial tasks found
                break
        
        # Cascade Update
        for tid in self.tasks_sequence_list:
            self.significance_list[tid] = self._calculate_significance(self.all_tasks[tid])
