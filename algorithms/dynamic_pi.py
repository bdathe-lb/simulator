# --------------------------------------------------------
# File: algorithms/dynamic_pi.py
#
# "Robust Dynamic PI for Communication-Constrained Environments"
#
# Key Academic Mechanisms:
#   1. Cyber-Physical Binding (Execution Locking): 
#      Resolves assignment-execution conflicts by prioritizing physical states.
#   2. Scalar Task Consensus (Lightweight Consistency): 
#      Replaces vector clocks with task-based scalar timestamps for robustness.
#   3. Inertia-Based Stability Control: 
#      Prevents "ping-pong" effects in dynamic topologies.
#   4. Tiered Spatio-Temporal Inclusion: 
#      Heuristically prioritizes tasks by urgency (deadline) then proximity.
# --------------------------------------------------------

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from algorithms.base import BaseAlgorithm
from core.task import Task

# --- Constants Definitions ---
# Represents a task that is unassigned or has infinite cost
SIG_INFINITY = float('inf')   
# Represents a task that is physically being executed and cannot be preempted
SIG_HARD_LOCK = -1.0          

class DynamicPI(BaseAlgorithm):
    """
    Implements the Robust Dynamic PI Algorithm.

    This algorithm extends the Performance Impact (PI) method to handle 
    dynamic environments with restricted communication. It introduces a 
    Cyber-Physical state machine to handle execution conflicts and uses 
    scalar consensus for robust consistency.
    """

    def __init__(self, agent_id: int, max_tasks: int, all_tasks: List[Task]) -> None:
        """
        Initializes the Dynamic PI algorithm state.

        Args:
            agent_id: The unique ID of the agent.
            max_tasks: The maximum number of tasks this agent can carry.
            all_tasks: Reference list of all available tasks.
        """
        super().__init__(agent_id, max_tasks, all_tasks)
        num_tasks = len(all_tasks)
        
        # =========================================================
        # 1. Core Algorithm State
        # =========================================================
        # The planned execution sequence of task IDs
        self.tasks_sequence_list: List[int] = []
        # The significance list (Gamma).
        # Values can be:
        #   - SIG_HARD_LOCK: Physically locked by an agent
        #   - SIG_INFINITY: Unassigned
        #   - Float value: The estimated removal cost
        self.significance_list: List[float] = [SIG_INFINITY] * num_tasks
        # The assigned agent list
        self.assigned_agent_list: List[int] = [-1] * num_tasks 
        
        # =========================================================
        # 2. Consensus State
        # =========================================================
        # Scalar Timestamp List
        # Records the last update time for each task to resolve conflicts
        self.task_timestamps: List[float] = [0.0] * num_tasks
        
        # =========================================================
        # 3. Stability Control State
        # =========================================================
        # Cooling Blacklist: {task_id: expiration_time}
        # Prevents the agent from immediately re-bidding on a lost task
        self.blacklist: Dict[int, float] = {}
        
        # =========================================================
        # 4. Context Awareness & Binding
        # =========================================================
        self.current_time = 0.0
        self._is_currently_congested = False

        self.agent = None            # Bound in bind_agent()
        self.feasibility_matrix = [] # Bound in finalize_setup()

        # =========================================================
        # 5. Hyperparameters
        # =========================================================
        self.BLACKLIST_DURATION = 10.0      # Duration to ignore a lost task
        self.INERTIA_THRESHOLD = 15.0       # Min improvement needed to switch owners
        self.STALE_TIMEOUT = 30.0           # Time before resetting stale info
        self.CONGESTION_PENALTY = 2000.0    # Significance penalty during congestion
        self.CONGESTION_RADIUS = 20.0       # Radius to detect neighbor congestion

    # *******************************************************
    # Interface Implementation & State Mapping
    # *******************************************************
    
    def bind_agent(self, agent: Any) -> None:
        """Binds the physical agent instance to the algorithm."""
        self.agent = agent

    def finalize_setup(self, feasibility_matrix: List[List[bool]]) -> None:
        """Sets up the global feasibility matrix."""
        self.feasibility_matrix = feasibility_matrix
        
    def get_plan(self) -> List[int]:
        """Returns the current planned task sequence."""
        return self.tasks_sequence_list

    def pack_message(self) -> Dict[str, Any]:
        """
        Packs the internal state into a message for broadcasting.

        [Mechanism: Cyber-Physical Binding]
        If the agent is physically executing a task, it overrides the broadcast 
        significance to SIG_HARD_LOCK, regardless of internal calculations.

        Returns:
            A dictionary containing the agent's state, bids, and timestamps.
        """
        assert self.agent is not None, "Agent must be bound" 
       

        # 1. Copy the current Significance list
        broadcast_significance = list(self.significance_list)
        
        # 2. Enforce Hard Lock for physical consistency
        # If I am physically executing a task, I must broadcast a Hard Lock.
        if self.agent.status == 'EXECUTING' and self.agent.current_target_task_id is not None:
            target_task_id = self.agent.current_target_task_id
            broadcast_significance[target_task_id] = SIG_HARD_LOCK
            
            # Update timestamp to current time to ensure neighbors accept this lock
            self.task_timestamps[target_task_id] = self.current_time
        
        # 3. Construct message packet
        return {
            "sender_id": self.agent_id,
            "significance_list": broadcast_significance,
            "assigned_agent_list": self.assigned_agent_list,
            "task_timestamps": self.task_timestamps,
            "sender_position": self.agent.position
        }

    # *******************************************************
    # Cyber-Physical Event Hooks
    # *******************************************************

    def on_task_locked(self, task_id: int) -> None:
        """
        Triggered when the physical agent starts executing a task.
        
        [Mechanism: Execution Locking]
        Transitions the task state from 'Soft Assignment' (significance-based) to 
        'Hard Lock' (physical authority), preventing preemption.

        Args:
            task_id: The ID of the task being locked.
        """
        # 1. Force-set Hard Lock signal
        self.significance_list[task_id] = SIG_HARD_LOCK
        
        # 2. Confirm ownership locally
        self.assigned_agent_list[task_id] = self.agent_id
        
        # 3. Update timestamp to prioritize this state
        self.task_timestamps[task_id] = self.current_time

    def on_task_completed(self, task_id: int) -> None:
        """
        Triggered when the task is physically completed.
        
        Cleans up the internal state to prevent the algorithm from 
        re-assigning the completed task.

        Args:
            task_id: The ID of the completed task.
        """
        # 1. Remove from the execution queue
        if task_id in self.tasks_sequence_list:
            self.tasks_sequence_list.remove(task_id)
            
        # 2. Reset algorithm state for this task
        self.significance_list[task_id] = SIG_INFINITY
        self.assigned_agent_list[task_id] = -1

    # *******************************************************
    # Main Control Loop
    # *******************************************************

    def run_iteration(self, messages: List[Dict[str, Any]]) -> None:
        """
        Executes one iteration of the dynamic control loop.

        This follows a structured cycle: 
        Sense (Context) -> Maintain -> Perceive (Consensus) -> Plan (Removal/Inclusion).

        Args:
            messages: List of messages received from neighbors.
        """
        assert self.agent is not None, "Agent must be bound to run iteration" 
        self.current_time += 1.0

        # 1. Context Awareness: Check for local congestion
        self._is_currently_congested = self._calculate_congestion_status(messages)
        
        # 2. Maintenance: Clear stale information
        self._check_stale_information()
        
        # 3. Consensus Phase: Resolve conflicts using Scalar Consensus
        self._consensus_phase(messages)
        
        # 4. Planning Phase A: Task Removal
        self._task_removal_phase()
        
        # 5. Planning Phase B: Task Inclusion
        self._task_inclusion_phase()
        
        # 6. Cleanup: Manage cooling timers
        self._cleanup_blacklist()

    # *******************************************************
    # Phase 1: Consensus
    # *******************************************************

    def _consensus_phase(self, messages: List[Dict[str, Any]]) -> None:
        """
        Phase 1: Scalar Task Consensus.

        [Mechanism: Scalar Consensus with Inertia & Hard Locks]
        Resolves conflicts using task-based scalar timestamps instead of vector clocks.
        It prioritizes physical execution states (Hard Locks) and uses an inertia 
        threshold to prevent system oscillation ("ping-pong" effect).

        Args:
            messages: List of messages received from neighbors.
        """
        assert self.agent is not None

        for msg in messages:
            neighbor_significance = msg['significance_list']
            neighbor_assigned = msg['assigned_agent_list']
            neighbor_timestamps = msg['task_timestamps']

            for task_id in range(len(self.all_tasks)):
                # --- Rule 1: Self-Execution Authority ---
                # If I am physically executing the task, I am the ultimate authority.
                # I ignore neighbor info and refresh my timestamp.
                if self.agent.status == 'EXECUTING' and self.agent.current_target_task_id == task_id:
                    self.task_timestamps[task_id] = self.current_time
                    self.significance_list[task_id] = SIG_HARD_LOCK
                    self.assigned_agent_list[task_id] = self.agent_id
                    continue

                # --- Rule 2: Neighbor Hard Lock Authority ---
                # If neighbor signals a Hard Lock, they win immediately.
                if neighbor_significance[task_id] == SIG_HARD_LOCK:
                    self._accept_neighbor_info(task_id, neighbor_significance, neighbor_assigned, neighbor_timestamps)
                    
                    # If I thought I owned it, I lost it. Enter cooling period.
                    if self.assigned_agent_list[task_id] == self.agent_id:
                         self.blacklist[task_id] = self.current_time + self.BLACKLIST_DURATION
                    continue
                
                # --- Rule 3: Scalar Freshness & Inertia ---
                my_ts = self.task_timestamps[task_id]
                neighbor_ts = neighbor_timestamps[task_id]
                
                # Case A: Neighbor has strictly newer information
                if neighbor_ts > my_ts:
                    if self.assigned_agent_list[task_id] == self.agent_id:
                        # [Mechanism: Inertia]
                        # If I own the task, only switch if neighbor's significance (cost) 
                        # is significantly lower than mine (by INERTIA_THRESHOLD).
                        improvement = self.significance_list[task_id] - neighbor_significance[task_id]
                        if improvement > self.INERTIA_THRESHOLD:
                            self._accept_neighbor_info(task_id, neighbor_significance, neighbor_assigned, neighbor_timestamps)
                        else:
                            # Reject the switch, but update timestamp to acknowledge I saw the bid.
                            # This prevents the neighbor from repeatedly sending the same "new" info.
                            self.task_timestamps[task_id] = self.current_time
                    else:
                        # I am not the owner, simply accept the newer information.
                        self._accept_neighbor_info(task_id, neighbor_significance, neighbor_assigned, neighbor_timestamps)

                # Case B: Concurrent information (Approximate Equal timestamps)
                elif math.isclose(neighbor_ts, my_ts):
                    # Tie-Breaker: Lower Significance wins
                    if neighbor_significance[task_id] < self.significance_list[task_id]:
                         self._accept_neighbor_info(task_id, neighbor_significance, neighbor_assigned, neighbor_timestamps)
                    # Tie-Breaker: If Significance equal, Lower Agent ID wins
                    elif math.isclose(neighbor_significance[task_id], self.significance_list[task_id]):
                        if neighbor_assigned[task_id] != -1 and neighbor_assigned[task_id] < self.assigned_agent_list[task_id]:
                             self._accept_neighbor_info(task_id, neighbor_significance, neighbor_assigned, neighbor_timestamps)

    def _accept_neighbor_info(self, task_id, n_sigs, n_owners, n_timestamps):
        """Helper to atomically update local state with neighbor's info."""
        self.significance_list[task_id] = n_sigs[task_id]
        self.assigned_agent_list[task_id] = n_owners[task_id]
        self.task_timestamps[task_id] = n_timestamps[task_id]

    # *******************************************************
    # Phase 2: Removal 
    # *******************************************************

    def _task_removal_phase(self) -> None:
        """
        Phase 2: Task Removal.

        Removes tasks that were lost during consensus or have become 
        inefficient due to congestion or constraint violations.
        """
        assert self.agent is not None
        
        # 1. Remove tasks lost during Consensus (Ownership mismatch)
        lost_tasks = [
            task_id for task_id in self.tasks_sequence_list
            if self.assigned_agent_list[task_id] != self.agent_id
        ]
        for task_id in lost_tasks:
            self.tasks_sequence_list.remove(task_id)
            self.blacklist[task_id] = self.current_time + self.BLACKLIST_DURATION
            
        # 2. Iterative Greedy Removal (Synergy & Efficiency Check)
        while True:
            max_reduction = -SIG_INFINITY
            task_to_remove = -1
            
            # [Mechanism: Congestion Avoidance]
            # Reduce willingness to keep tasks if in a congested area.
            penalty = self.CONGESTION_PENALTY if self._is_currently_congested else 0.0
            
            for task_id in self.tasks_sequence_list:
                # Never remove the task currently being executed (Physical Hard Lock)
                if task_id == self.agent.current_target_task_id and self.agent.status == 'EXECUTING':
                    continue
                
                local_significance = self._calculate_significance(task_id)
                
                # Adjust local value with penalty
                adjusted_local_sig = local_significance + penalty 
                global_sig = self.significance_list[task_id]
                
                # Check 1: Constraint Violation (Deadline missed -> Infinite Cost)
                # This task will cause a timeout and needs to be removed
                if math.isinf(local_significance):
                    diff = SIG_INFINITY
                # Check 2: Efficiency (Am I still the best agent for this task?)
                # If (My Adjusted Cost) > (Market Cost), I am inefficient.
                else:
                    diff = adjusted_local_sig - global_sig

                if diff > max_reduction:
                    max_reduction = diff
                    task_to_remove = task_id
            
            # Execute removal if it improves the objective (reduces cost discrepancy)
            if max_reduction > 1e-6:
                self.tasks_sequence_list.remove(task_to_remove)
                # Note: No re-optimization call here as requested
            else:
                break

    # *******************************************************
    # Phase 3: Inclusion 
    # *******************************************************

    def _task_inclusion_phase(self) -> None:
        """
        Phase 3: Tiered Spatio-Temporal Inclusion.

        [Mechanism: Tiered Spatio-Temporal Inclusion]
        Heuristically selects tasks to add. It departs from pure greedy selection 
        by prioritizing tasks based on 'Urgency' (Deadline) first, and then 
        'Proximity' (Distance).
        """
        assert self.agent is not None
        
        # Capacity check
        if len(self.tasks_sequence_list) >= self.max_tasks:
            return

        # 1. Filter Candidates
        candidates = [
            task for task in self.all_tasks 
            if not task.completed 
            and task.id not in self.blacklist 
            and task.id not in self.tasks_sequence_list 
            and self.feasibility_matrix[self.agent_id][task.id]
        ]
        
        # 2. Tiered Sorting Strategy
        URGENCY_THRESHOLD = 150.0 
        
        def tiered_key(task):
            """
            Sorts tasks into two tiers:
            Tier 1 (Critical): Deadline is approaching. Sort by time remaining.
            Tier 2 (Normal): Deadline is far. Sort by distance (Spatial).
            """
            assert self.agent is not None
            time_left = task.deadline - self.current_time
            
            if time_left < URGENCY_THRESHOLD:
                return (0, time_left) 
            else:
                dist = np.linalg.norm(task.position - self.agent.position)
                return (1, dist)
        
        candidates.sort(key=tiered_key)
        
        # 3. Greedy Insertion
        for task in candidates:
            # Re-check capacity
            if len(self.tasks_sequence_list) >= self.max_tasks:
                break
                
            marginal_significance, pos = self._calculate_marginal_significance(task)
            
            # If infeasible (deadline violation), skip
            if math.isinf(marginal_significance):
                continue
                
            global_significance = self.significance_list[task.id]
            significance_gain = global_significance - marginal_significance
            
            # If positive gain, acquire the task
            if significance_gain > 0:
                self.tasks_sequence_list.insert(pos, task.id)
                self.assigned_agent_list[task.id] = self.agent_id
                self.significance_list[task.id] = marginal_significance
                
                # Claim ownership by updating timestamp to now
                self.task_timestamps[task.id] = self.current_time 
                
        # Update significance for all tasks after modifications
        self._update_internal_significance()

    def _update_internal_significance(self) -> None:
        """Batch updates significance for all tasks in the local sequence."""
        for task_id in self.tasks_sequence_list:
            self.significance_list[task_id] = self._calculate_significance(task_id)

    # *******************************************************
    # Helpers & Math
    # *******************************************************

    def _calculate_congestion_status(self, messages: List[Dict]) -> bool:
        """
        Detects local congestion based on neighbor proximity.
        
        Returns:
            True if a higher-priority agent (lower ID) is within 
            CONGESTION_RADIUS, prompting a yield behavior.
        """
        assert self.agent is not None
        my_pos = self.agent.position
        
        for msg in messages:
            neighbor_pos = msg.get('sender_position')
            sender_id = msg['sender_id']
            if neighbor_pos is not None:
                dist = np.linalg.norm(my_pos - neighbor_pos)
                if dist < self.CONGESTION_RADIUS and sender_id < self.agent_id:
                    return True
        return False

    def _check_stale_information(self) -> None:
        """
        Identifies and resets expired task information.
        
        Prevents "Ghost Tasks" (tasks owned by agents who have left the 
        network or crashed) from blocking assignment indefinitely.
        """
        for task_id in range(len(self.all_tasks)):
            if self.all_tasks[task_id].completed:
                continue
            if self.assigned_agent_list[task_id] == self.agent_id: 
                continue
            if self.assigned_agent_list[task_id] == -1: 
                continue
            
            # If info is older than threshold, reset to unassigned
            if (self.current_time - self.task_timestamps[task_id]) > self.STALE_TIMEOUT:
                self.assigned_agent_list[task_id] = -1
                self.significance_list[task_id] = SIG_INFINITY
                self.task_timestamps[task_id] = self.current_time

    def _cleanup_blacklist(self) -> None:
        """Removes expired entries from the blacklist."""
        expired_tasks = [tid for tid, expiry in self.blacklist.items() if self.current_time >= expiry]
        for tid in expired_tasks:
            del self.blacklist[tid]

    # --- Core Cost Calculations (Physical Layer) ---

    def _calculate_total_path_cost(self, path: List[int]) -> float:
        """
        Calculates the total time cost for a task sequence starting from 
        the agent's *current* dynamic position.
        
        Returns:
            Total time in seconds. Returns SIG_INFINITY if infeasible.
        """
        if self.agent is None: 
            return SIG_INFINITY
        cost = 0.0
        curr_pos = self.agent.position
        current_arrival_time = self.current_time 
        
        for task_id in path:
            task = self.all_tasks[task_id]
            dist = np.linalg.norm(task.position - curr_pos)
            travel_time = dist / self.agent.speed
            
            arrival_at_task = current_arrival_time + travel_time
            if arrival_at_task > task.deadline:
                return SIG_INFINITY
            
            cost += travel_time + task.exec_duration
            current_arrival_time = arrival_at_task + task.exec_duration
            curr_pos = task.position
        return cost

    def _calculate_significance(self, task_id: int) -> float:
        """
        Calculates the significance (marginal cost contribution) of a task.
        Significance = Cost(Path) - Cost(Path without task).
        """
        if task_id not in self.tasks_sequence_list: 
            return SIG_INFINITY
        
        cost_with = self._calculate_total_path_cost(self.tasks_sequence_list)
        if math.isinf(cost_with): 
            return SIG_INFINITY
            
        temp_sequence = self.tasks_sequence_list.copy()
        temp_sequence.remove(task_id)
        cost_without = self._calculate_total_path_cost(temp_sequence)
        
        return cost_with - cost_without

    def _calculate_marginal_significance(self, task: Task) -> Tuple[float, int]:
        """
        Calculates the marginal significance (insertion cost) of adding a new task.
        Returns the minimum cost increase and the best insertion index.
        """
        assert self.agent is not None

        min_marginal = SIG_INFINITY
        best_pos = -1
        base_cost = self._calculate_total_path_cost(self.tasks_sequence_list)
        
        if math.isinf(base_cost) and len(self.tasks_sequence_list) > 0:
             return SIG_INFINITY, -1
        
        for i in range(len(self.tasks_sequence_list) + 1):
            # [Hard Lock Protection] Cannot insert at index 0 if executing
            if i == 0 and self.agent.status == 'EXECUTING' and len(self.tasks_sequence_list) > 0:
                continue
            
            temp_sequence = self.tasks_sequence_list.copy()
            temp_sequence.insert(i, task.id)
            new_cost = self._calculate_total_path_cost(temp_sequence)
            
            if math.isinf(new_cost): 
                continue
                
            marginal = new_cost - base_cost
            if marginal < min_marginal:
                min_marginal = marginal
                best_pos = i
        return min_marginal, best_pos
