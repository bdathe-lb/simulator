# --------------------------------------------------------
# File: core/environment.py
# The environment for executing the simulation includes 
# agents, tasks, topology, and other components.
# --------------------------------------------------------

# The feature implementation is correct fine.

import numpy as np
from typing import Dict, List, Iterator, Any
from collections import defaultdict
from core.agent import Agent
from core.task import Task
from core.topology import Topology

# Additional rounds of information transmission
CONSENSUS_HOPS = 3

class Environment:
    """
    Simulation environment managing agents, tasks, and the simulation loop.

    This class acts as the central controller for the multi-agent system simulation.
    It handles the initialization of entities, updates the network topology based on
    agent positions, and drives the main simulation loops for both static and
    dynamic allocation modes.

    Attributes:
        agents (List[Agent]): A list of all agent instances in the simulation.
        tasks (List[Task]): A list of all task instances to be assigned or executed.
        topology (Topology): The communication graph representing connectivity between agents.
        communication_radius (float): The maximum distance between two agents to establish
            a communication link.
        fixed_topology (bool): If True, the topology will not be updated based on distance.
    """
    def __init__(self, agents: List[Agent], tasks: List[Task],
                 topology: Topology, communication_radius: float,
                 fixed_topology: bool = False) -> None:
        """
        Initializes the simulation environment.

        Args:
            agents: A list of Agent objects.
            tasks: A list of Task objects.
            topology: A Topology object managing the communication graph.
            communication_radius: The maximum distance between agents to allow communication.
            fixed_topology: If True, the topology will not be updated based on distance.
        """
        self.agents = agents
        self.tasks = tasks
        self.topology = topology
        self.communication_radius = communication_radius
        self.fixed_topology = fixed_topology  

    # *****************************************
    # Mode 1: Dynamic Simulation (For V1, V2)
    # *****************************************
    def run_dynamic_simulation(self, dt: float, max_time: float) -> Iterator[Dict[str, Any]]:
        """
        Runs a unified simulation loop where allocation and execution happen simultaneously.

        This mode simulates a dynamic environment where agents move, communicate,
        and execute tasks in real-time. It is suitable for algorithms like V1 and V2
        that support dynamic reallocation.

        Args:
            dt: The time step duration for the simulation in seconds.
            max_time: The maximum duration to run the simulation in seconds.

        Yields:
            Dict[str, Any]: A dictionary containing the current snapshot of the system state.
                Keys include 'agents', 'tasks', 'topology', 'time', and 'iteration'.
        """
        # Current simulation time
        current_time = 0.0

        # Ensure initial 'IDLE' state
        for agent in self.agents:
            agent.status = 'IDLE'

        # Initializes communication topology
        self._update_dynamic_topology()

        # Loop
        while current_time < max_time:
            # --- 1. Update topology ---
            self._update_dynamic_topology()

            # --- 2. Run algorithm ---
            # Let the information circulate a little longer
            for _ in range(CONSENSUS_HOPS):
                self._run_communication_step()

            # --- 3. Physics update ---
            all_idle = True
            for agent in self.agents:
                agent.update_state(dt, self.tasks)
                if not agent.is_idle():
                    all_idle = False
            
            # --- 4. Termination check ---
            incomplete_tasks = [t for t in self.tasks if not t.completed]
            
            # Case A: All tasks are completed, and all agents are in an idle state
            if not incomplete_tasks:
                if all_idle:
                    print(f"\n[Success] All tasks completed at {current_time:.2f}s")
                    self._print_final_log()
                    yield self.get_current_state(current_time)
                    break
            # Case B: All agents are in an idle state, but there are unfinished tasks
            elif all_idle:
                # If everyone is idle but tasks remain, it's a deadlock (impossible tasks)
                print(f"\n[Terminated] Deadlock detected at {current_time:.2f}s. "
                      f"{len(incomplete_tasks)} tasks remaining.")
                self._print_final_log()
                yield self.get_current_state(current_time)
                break

            # --- 5. Yield state for amination ---
            yield self.get_current_state(current_time)
            current_time += dt

    # *****************************************
    # Mode 2: Static Allocation (For PI - Phase 1)
    # *****************************************
    def run_static_allocation(self, max_iterations: int, convergence_threshold: int = 10) -> Iterator[Dict[str, Any]]:
        """
        Runs only the allocation algorithm without physical movement (Phase 1).

        This mode is specific to the original PI algorithm's first phase, where agents
        negotiate task assignments until consensus is reached or the iteration limit is hit.
        Agents do not move during this phase.

        Args:
            max_iterations: The maximum number of algorithm iterations to run.
            convergence_threshold: The number of consecutive stable iterations required
                to consider the allocation converged.

        Yields:
            Dict[str, Any]: A dictionary containing the current snapshot of the system state
                for visualization.
        """
        print("--- Phase 1: Static Task Allocation ---")

        # Initial topology calculation (Static)
        if self.topology.graph.number_of_edges() == 0:
             self._update_dynamic_topology()

        last_plan_signature = None
        stable_count = 0

        for i in range(max_iterations):
            # Run communication and decision making
            self._run_communication_step()

            # Check convergence
            # We construct a signature of all agents' plans to detect stability
            current_plans = tuple(tuple(a.algorithm.get_plan()) for a in self.agents)
            
            if current_plans == last_plan_signature:
                stable_count += 1
            else:
                stable_count = 0
            last_plan_signature = current_plans

            yield self.get_current_state(iteration=i)

            if stable_count >= convergence_threshold:
                print(f"Allocation converged after {i+1} iterations.")
                break
        else:
            print(f"Allocation stopped after max {max_iterations} iterations.")

    # *****************************************
    # Mode 2: Static Execution (For PI - Phase 2)
    # *****************************************
    def run_static_execution(self, dt: float, max_time: float) -> Iterator[Dict[str, Any]]:
        """
        Runs physical execution based on pre-calculated plans (Phase 2).

        This mode follows the static allocation phase. Agents execute their assigned
        tasks without further negotiation. The simulation ends when all reachable tasks
        are completed.

        Args:
            dt: The time step duration for the simulation in seconds.
            max_time: The maximum duration to run the simulation in seconds.

        Yields:
            Dict[str, Any]: A dictionary containing the current snapshot of the system state.
        """
        print("\n--- Phase 2: Mission Execution ---")
        current_time = 0.0

        # Loop
        while current_time < max_time:
            # Physics update only
            all_agents_idle = True
            for agent in self.agents:
                agent.update_state(dt, self.tasks)
                if not agent.is_idle():
                    all_agents_idle = False

            # Check if finished
            if all_agents_idle:
                completed_count = sum(1 for t in self.tasks if t.completed)
                total_count = len(self.tasks)
                if completed_count == total_count:
                    print(f"[Success] All tasks completed at {current_time:.2f}s")
                else:
                    print(f"\n[Terminated] Deadlock detected at {current_time:.2f}s. "
                      f"{total_count - completed_count}  tasks remaining.")

                # Print informations
                self._print_final_log()
                yield self.get_current_state(current_time)
                break
            # Yields state for animation
            yield self.get_current_state(current_time)
            current_time += dt
    
    # *****************************************
    # Helpers
    # *****************************************
    def _run_communication_step(self) -> None:
        """
        Executes one round of message exchange and algorithm updates.

        This helper method performs three steps:
        1. Collects messages prepared by all agents.
        2. Routes messages to neighbors based on the current topology.
        3. Triggers the algorithm step for each agent to process received messages.
        """
        # --- 1. Prepare messages for neighbors ---
        prepare_msgs = [agent.prepare_message() for agent in self.agents]
        
        # --- 2. Route messages to neighbors ---
        inbox = defaultdict(list)
        for sender in self.agents:
            neighbors = self.topology.get_neighbors(sender.id)
            for nid in neighbors:
                inbox[nid].append(prepare_msgs[sender.id])

        # --- 3. Process messages from neighbors ---
        for agent in self.agents:
            agent.run_algorithm_step(inbox[agent.id])

    def _update_dynamic_topology(self) -> None:
        """
        Recalculates the communication graph based on current agent positions.

        It computes the pairwise distances between all agents. If the distance
        is within `communication_radius`, a link is added to the adjacency matrix.
        The topology object is then updated with this new matrix.

        If `fixed_topology` is True, this method does nothing, preserving the
        manually initialized network structure.
        """
        # For original PI
        if self.fixed_topology:
            return

        # We delegate the graph building to the Topology class, 
        # but we need to provide the adjacency matrix logic here
        num_agents = len(self.agents)
        adj_matrix = [[False] * num_agents for _ in range(num_agents)]
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(self.agents[i].position - self.agents[j].position)
                if dist <= self.communication_radius:
                    adj_matrix[i][j] = True
                    adj_matrix[j][i] = True

        self.topology.update_from_adjacency_matrix(adj_matrix)

    def get_current_state(self, time: float = 0.0, iteration: int = 0) -> Dict[str, Any]:
        """
        Captures the current snapshot of the simulation state.

        Args:
            time: The current simulation timestamp.
            iteration: The current algorithm iteration count (for static mode).

        Returns:
            Dict[str, Any]: A dictionary containing references to agents, tasks,
                topology, and current time/iteration metadata for visualization.
        """
        return {
            "agents": self.agents,
            "tasks": self.tasks,
            "topology": self.topology,
            "time": time,
            "iteration": iteration
        }

    def _print_final_log(self):
        """
        Prints a summary log of tasks completed by each agent to the console.
        """
        print("-" * 40)
        print("Final Execution Log:")
        for agent in self.agents:
            print(f"Agent {agent.id} ({agent.agent_type}) completed: {agent.completed_tasks_log}")
        print("-" * 40)
