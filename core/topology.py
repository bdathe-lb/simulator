# --------------------------------------------------------
# File: algorithms/topology.py
# Implement network communication in the simulation using 
# the NetworkX library to achieve dynamic network changes.
# --------------------------------------------------------

# The feature implementation is correct fine.

import networkx as nx
from typing import List

class Topology:
    """
    Manages the communication topology between agents using a graph structure.

    Attributes:
        graph (nx.Graph): NetworkX graph representing connectivity.
    """

    def __init__(self, agent_ids: List[int]) -> None:
        """
        Initializes the topology with a set of agent nodes.

        Args:
            agent_ids: A list of agent IDs to be added as nodes.
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(agent_ids)

    def update_from_adjacency_matrix(self, adj_matrix: List[List[bool]]) -> None:
        """
        Rebuilds the graph edges based on an adjacency matrix.

        Args:
            adj_matrix: A square matrix where adj_matrix[i][j] is True if
                        agents i and j are connected.
        """
        self.graph.clear_edges()
        num_agents = len(adj_matrix)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if adj_matrix[i][j]:
                    self.graph.add_edge(i, j)

    def is_connected(self) -> bool:
        """
        Checks if the entire graph is connected.

        Returns:
            True if the graph is connected, False otherwise.
        """
        if self.graph.number_of_nodes() == 0:
            return True
        return nx.is_connected(self.graph)
    
    def get_agent_count(self) -> int:
        """
        Returns the total number of agents in the topology.
        """
        return self.graph.number_of_nodes()

    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Retrieves the IDs of neighbors connected to a specific agent.

        Args:
            agent_id: The ID of the agent query.

        Returns:
            A list of neighbor agent IDs. Returns an empty list if the agent
            is not in the graph.
        """
        if agent_id not in self.graph:
            return []
        return list(self.graph.neighbors(agent_id))
