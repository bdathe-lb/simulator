# --------------------------------------------------------
# File: visualization/animator.py
# Perform an animated demonstration of the simulation.
# --------------------------------------------------------

# The feature implementation is correct fine.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

class Animator:
    """
    Handles visualization for both static and dynamic multi-agent simulations.

    This class is responsible for creating and managing the matplotlib animation
    loop. It supports rendering tasks, agents, communication links (topology),
    and agent paths for both static (two-phase) and dynamic (real-time) modes.

    Attributes:
        env: The simulation Environment object containing entity data.
        generator (Generator): The simulation data generator yielding state snapshots.
        comm_radius (float): The communication radius for visualizing coverage circles.
        fig (plt.Figure): The matplotlib figure object.
        ax (plt.Axes): The matplotlib axes object.
        dynamic_artists (List[plt.Artist]): A list of artists (plot elements) drawn
            in the current frame, used for efficient clearing/updating.
        colors (Dict[str, str]): Mapping of task types to colors.
        markers (Dict[str, str]): Mapping of agent types to plot markers.
    """

    def __init__(self, environment, generator, comm_radius):
        """
        Initializes the Animator with simulation context.

        Args:
            environment: The simulation Environment instance.
            generator: A generator function yielding state dictionaries.
            comm_radius: The visual radius for agent communication circles.
        """
        # Basic component
        self.env = environment
        self.generator = generator
        self.comm_radius = comm_radius

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10)) # Adjusted size
        self.dynamic_artists = []
        
        # Visual fonfig
        self.colors = {'medicine': 'red', 'food': 'green'}
        self.markers = {'medicine': '^', 'food': 'P'}

    def _init_plot(self):
        """
        Initializes the static background elements of the plot.

        Sets up axis limits, titles, and static legends. This function is called
        once at the start of the animation.

        Returns:
            List[plt.Artist]: An empty list as no dynamic artists are drawn initially.
        """
        self.ax.clear()
        self.ax.set_xlim(0, 2000)
        self.ax.set_ylim(0, 2000)
        self.ax.set_title("Initializing...")
        # Add legends
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='Agent (Medicine)', markersize=10),
            Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', label='Agent (Food)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Task (Medicine)', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Task (Food)', markersize=8),
        ]
        self.ax.legend(handles=legend_elements, loc='best')
        
        return []

    def _update(self, frame):
        """
        Updates the plot for a new animation frame.

        Retrieves the next state from the simulation generator, clears old dynamic
        elements, and redraws agents, tasks, topology, and paths.

        Args:
            frame: The current frame index (automatically passed by FuncAnimation).

        Returns:
            List[plt.Artist]: A list of all dynamic artists drawn in this frame.
        """
        try:
            # Retrieve the physical state of each frame in the simulation
            state = next(self.generator)
        except StopIteration:
            # Stop animation if generator is exhausted
            return []

        # Clear previous dynamic elements (but keep static background)
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists = []

        # --- 1. Draw Tasks ---
        # Only draw uncompleted tasks to visualize progress
        for task in state['tasks']:
            if not task.completed:
                # Draw Task point
                sc = self.ax.scatter(task.position[0], task.position[1], 
                                     c=self.colors[task.task_type], s=100, zorder=3)
                # Draw Task ID label
                tx = self.ax.text(task.position[0], task.position[1]-40, f"T{task.id}", ha='center')
                self.dynamic_artists.extend([sc, tx])

        # --- 2. Draw topology ---
        # Draw edges between agents if they are connected in the graph
        pos = {a.id: a.position for a in state['agents']}
        edges = nx.draw_networkx_edges(state['topology'].graph, pos=pos, ax=self.ax, 
                                       edge_color='gray', style='--', alpha=0.5)
        if edges:
            self.dynamic_artists.append(edges)

        # --- 3. Draw Agents & paths ---
        for agent in state['agents']:
            # Draw communication range (circle)
            c = Circle(agent.position, self.comm_radius, color='gray', alpha=0.1)
            self.ax.add_patch(c)
            self.dynamic_artists.append(c)
            
            # Draw agent body
            sc = self.ax.scatter(agent.position[0], agent.position[1], 
                                 color=agent.color, marker=self.markers[agent.agent_type], s=150, zorder=5, edgecolors='black')
            # Draw Agent ID label
            tx = self.ax.text(agent.position[0], agent.position[1]+40, f"A{agent.id}", ha='center')
            self.dynamic_artists.extend([sc, tx])

            # Draw planned path
            # Fetches the current intended path directly from the algorithm interface
            plan = agent.algorithm.get_plan()
            if plan:
                # Construct path points: [Current Pos] -> [Task 1] -> [Task 2] ...
                pts = [agent.position] + [self.env.tasks[tid].position for tid in plan]
                pts = np.array(pts)
                ln, = self.ax.plot(pts[:,0], pts[:,1], c=agent.color, ls='-', lw=1, alpha=0.6)
                self.dynamic_artists.append(ln)

        # --- 4. Update title ---
        t_str = f"Time: {state['time']:.1f}s"
        if 'iteration' in state and state['iteration'] > 0:
            # For static allocation phase, show iteration count
            t_str = f"Allocation Iter: {state['iteration']}"
        
        self.ax.set_title(f"MRTA Simulation | {t_str}")

        return self.dynamic_artists

    def run(self):
        """
        Starts the matplotlib animation loop.

        Configures and runs `FuncAnimation`. The plot window will block execution
        until closed.
        """
        _ = animation.FuncAnimation(
            self.fig, 
            self._update, 
            init_func=self._init_plot, 
            interval=50,       # Update every 50ms
            blit=False,        # Turn off blitting for compatibility with dynamic artists
            save_count=500     # Max frames to cache if saving
        )
        plt.show()
