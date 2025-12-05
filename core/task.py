# --------------------------------------------------------
# File: core/task.py
# Implement the Task entity in the simulation.
# --------------------------------------------------------

# The feature implementation is correct fine.

import numpy as np

# Define task type constants
TASK_TYPE_MEDICINE = 'medicine'
TASK_TYPE_FOOD     = 'food'

class Task:
    """
    Represents a task in the simulation enviroment.

    Attributes:
        id (int): Unique identifier for the task.
        task_type (str): Type of the task (e.g., 'medicine', 'food').
        position (np.ndarray): The (x, y) coordinates of the task.
        deadline (float): The simulation time by which the task must be started.
        exec_duration (float): The time required to complete the task once started.
        completed (bool): Status flag indicating if the task has been finished.
    """

    def __init__(self, id: int, task_type: str, position: tuple,
                 deadline: float, exec_duration: float) -> None:
        """
        Initializes a Task instance.

        Args:
            id: Unique identifier for the task.
            task_type: Type string constant.
            position: Tuple (x, y) for spatial coordinates.
            deadline: Deadline timestamp.
            exec_duration: Duration required to execute the task.
        """
        self.id = id
        self.task_type = task_type
        self.position = np.array(position, dtype=float)
        self.deadline = deadline
        self.exec_duration = exec_duration
        self.completed = False

    def __repr__(self) -> str:
        pos_str = np.array2string(self.position, separator=',')
        return (f"Task(id={self.id}, type='{self.task_type}', "
                f"pos={pos_str}, deadline={self.deadline}, "
                f"duration={self.exec_duration}, completed={self.completed})")
    
    def __str__(self) -> str:
        return (f"Task #{self.id} ({self.task_type}) "
                f"@ {self.position}, due={self.deadline:.1f}")
