# --------------------------------------------------------
# File: core/base.py
# Implement base classes for each algorithm, defining the 
# basic framework of the algorithms.
# --------------------------------------------------------

# The feature implementation is correct fine.

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.task import Task

class BaseAlgorithm(ABC):
    """
    Abstract base class for all task allocation algorithms.

    This class defines the standard interface that the Agent class uses to
    interact with decision-making logic. Any specific algorithm implementation
    (e.g., Original PI, Dynamic PI) must inherit from this class.

    Attributes:
        agent_id (int): ID of the agent owning this algorithm instance.
        max_tasks (int): Maximum number of tasks the agent can handle.
        all_tasks (List[Task]): Reference to the global task list.
    """
    
    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the base algorithm state.

        Args:
            agent_id: The ID of the agent.
            max_tasks: Capacity constraint for the agent.
            all_tasks: List of all available tasks in the environment.
        """
        self.agent_id = agent_id
        self.max_tasks = max_tasks
        self.all_tasks = all_tasks

        # Derived classed should Initialize their specific data structures here
        # e.g., `significance_list`, `timestamp_list`, etc.

    @abstractmethod
    def bind_agent(self, agent: Any) -> None:
        """
        Binds the physical agent instance to the algorithm.
        """
        self.agent = agent

    @abstractmethod
    def finalize_setup(self, feasibility_matrix: List[List[bool]]) -> None:
        """
        Performs second-stage initialization with global info.

        Args:
            feasibility_matrix: A matrix indicating which tasks are feasible
                                for which agents.
        """
        pass

    @abstractmethod
    def run_iteration(self, messages: List[Dict[str, Any]]) -> None:
        """
        Executes one iteration of the algorithm (e.g., inclusion, consensus, removal).

        Args:
            messages: A list of messages received from neighboring agents.
        """
        pass

    @abstractmethod
    def get_plan(self) -> List[int]:
        """
        Retrieves the current task execution sequence decided by the algorithm.

        Returns:
            A list of task IDs representing the planned path.
        """
        pass 

    @abstractmethod
    def pack_message(self) -> Dict[str, Any]:
        """
        Packs the internal state into a message dictionary for broadcasting.

        Returns:
            A dictionary containing relevant algorithm state (e.g., bids,
            timestamps) to be sent to neighbors.
        """
        pass

    # --- Hooks for Dynamic Events (Optional for Static Algos) ---
    def on_task_completed(self, task_id: int) -> None:
        """
        Hook called when the agent physically completes a task.

        Args:
            task_id: The ID of the completed task.
        """
        pass

    def on_task_locked(self, task_id: int) -> None:
        """
        Hook called when the agent physically starts executing a task (locks it).

        Args:
            task_id: The ID of the task being locked.
        """
        pass
