# --------------------------------------------------------
# File: core/agent.py
# Implemented the Agent entity in the simulation, 
# where each Agent is equipped with a decision-making 
# module for task assignment.
# --------------------------------------------------------

# The feature implementation is correct fine.

import numpy as np
from typing import Dict, List
from algorithms.base import BaseAlgorithm
from core.task import Task

# Define agent type constants
AGENT_TYPE_MEDICINE = 'medicine'
AGENT_TYPE_FOOD     = 'food'

class Agent:
    """
    Represents an agent in the simulation.

    The Agent is responsible for physical movement, state maintenance, and
    delegating decision-making to an injected algorithm strategy.

    Attributes:
        id (int): Unique identifier.
        agent_type (str): Type of the agent (e.g., 'medicine', 'food').
        position (np.ndarray): Current (x, y) coordinates.
        speed (float): Movement speed in units per second.
        color (Tuple): RGB color for visualization.
        algorithm (BaseAlgorithm): The decision-making strategy instance.
        status (str): Current physical status ('IDLE', 'MOVING', 'EXECUTING').
        completed_tasks_log (List[int]): History of completed task IDs.
    """

    def __init__(self, id: int, agent_type: str, position: tuple, speed: float,
                 color: tuple, algorithm: BaseAlgorithm) -> None:
        """
        Initializes the Agent.

        Args:
            id: Agent ID.
            agent_type: Type string.
            position: Initial (x, y) tuple.
            speed: Movement speed.
            color: Visualization color.
            algorithm: An instance of a subclass of BaseAlgorithm.
        """
        # Basic attribute
        self.id = id
        self.agent_type = agent_type
        self.position = np.array(position, dtype=float)
        self.speed = speed
        
        # Decision-Making module
        self.algorithm = algorithm
        self.algorithm.bind_agent(self)

        # Physical state
        self.color = color
        self.status = 'IDLE'
        self.current_target_task_id: int | None = None
        self.time_at_task = 0.0
        self.completed_tasks_log: List[int] = []

        # Statistic distance
        self.total_distance: float = 0.0

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, type={self.agent_type}, pos={self.position})"

    def update_state(self, dt: float, all_tasks: List[Task]) -> None:
        """
        Updates the agent's physical state for one time step.

        Moves towards the first task in the algorithm's plan. Handles task
        locking and completion logic.

        Args:
            dt: Time step duration.
            all_tasks: List of all Task objects (for position/status lookup).
        """
        # --- 1. Get current plan from the algorithm ---
        current_plan = self.algorithm.get_plan()

        # Filter out completed tasks from the plan
        # This prevent the agent from moving towards a tasks that is already done
        while current_plan:
            target_id = current_plan[0]
            if all_tasks[target_id].completed:
                # Notify algo to cleanup this stale task from its plan
                self.algorithm.on_task_completed(target_id)
                # Re-fetch plan after update
                current_plan = self.algorithm.get_plan()
                continue
            break

        # If no vaild tasks, become 'IDLE'
        if not current_plan:
            self.status = 'IDLE'
            self.current_target_task_id = None
            return
        
        # --- 2. Determine target task ---
        target_task_id = current_plan[0]
        target_task = all_tasks[target_task_id]
        self.current_target_task_id = target_task_id
        
        # --- 3. Movement logic ---
        dist_to_target = np.linalg.norm(target_task.position - self.position)
        move_dist = self.speed * dt

        # Case A: En route
        if dist_to_target > move_dist and self.status != 'EXECUTING':
            self.status = 'MOVING'
            # Calculate the direction vector (uint vector)
            direction = (target_task.position - self.position) / dist_to_target
            # Updates position infomation
            self.position += direction * move_dist
            # Cumulative moving distance is used for subsequent statistics
            self.total_distance += move_dist   

        # Case B: Arrived / Executing
        else:
            if self.status != 'EXECUTING':
                # Just arrived
                # Cumulative moving distance is used for subsequent statistics
                self.total_distance += move_dist    
                self.status = 'EXECUTING'
                self.position = target_task.position
                self.time_at_task = 0.0
                # Notify algo to lock this task (prevent others from taking it)
                self.algorithm.on_task_locked(target_task_id)

            # Execute task
            self.time_at_task += dt
            if self.time_at_task >= target_task.exec_duration:
                self._complete_task(target_task)
    
    def run_algorithm_step(self, messages: List[Dict]) -> None:
        """
        Delegates the decision-making step to the algorithm strategy.
        """
        self.algorithm.run_iteration(messages)

    def prepare_message(self) -> Dict:
        """
        Delegates message packaging to the algorithm strategy.
        """
        return self.algorithm.pack_message()

    def is_idle(self) -> bool:
        """
        Checks if the agent has no active tasks and is not executing.
        """
        return self.status == 'IDLE' and not self.algorithm.get_plan()

    def _complete_task(self, task: Task) -> None:
        """
        Internal handler for task completion.
        """
        if not task.completed:
            print(f"Agent {self.id} finished Task {task.id}")
            task.completed = True
            self.completed_tasks_log.append(task.id)
            # Notify algorithm to update internal state (remove task, update bids)
            self.algorithm.on_task_completed(task.id)

        self.status = 'IDLE'
        self.time_at_task = 0.0
