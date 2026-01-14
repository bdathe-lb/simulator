# --------------------------------------------------------
# File: algorithms/factory.py
# Implement using the factory pattern, defining an 
# interface for creating various specific algorithms.
# --------------------------------------------------------

# The feature implementation is correct fine.

from algorithms.base import BaseAlgorithm
from algorithms.original_pi import OriginalPI
from algorithms.dynamic_pi import DynamicPI
from typing import List
from core.task import Task

class AlgorithmFactory:
    """
    Factory class to instantiate algorithms based on a string key.
    """

    @staticmethod
    def create(algo_name: str, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> BaseAlgorithm:
        """
        Creates an instance of the requested algorithm strategy.

        Args:
            algo_name: Name key ('original', 'v1', 'v2', 'v3').
            agent_id: The agent's ID.
            max_tasks: Capacity constraint.
            all_tasks: Global task list.

        Returns:
            An initialized instance of a BaseAlgorithm subclass.
        """
        if algo_name == "original":
            return OriginalPI(agent_id, max_tasks, all_tasks)
        elif algo_name == "dynamic":
            return DynamicPI(agent_id, max_tasks, all_tasks)
        else:
            raise ValueError(f"Unknown algorithm name: {algo_name}")
