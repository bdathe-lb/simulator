#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TwoOptVisualizer:
    def __init__(self, num_cities=40, seed=42):
        """
        Initialize the TSP problem with random cities and set up the initial tour.
        
        Args:
            num_cities (int): Number of cities to generate.
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.num_cities = num_cities
        # Generate random coordinates (x, y) between 0 and 100
        self.cities = np.random.rand(num_cities, 2) * 100
        # Initial tour: simple sequence [0, 1, 2, ..., N-1]
        self.tour = np.arange(num_cities)
        # Calculate initial distance
        self.current_dist = self.calculate_total_distance(self.tour)
        # Generator for animation steps (yields improved tours)
        self.optimizer = self.two_opt_generator()

    def calculate_dist_matrix(self, c1, c2):
        """Calculate Euclidean distance between two points."""
        # Using vectorized operation for Euclidean distance
        return np.sqrt(np.sum((c1 - c2)**2))

    def calculate_total_distance(self, tour):
        """
        Calculate the total length of the current tour (including return to start).
        """
        dist = 0
        for i in range(len(tour)):
            # Connect current city to the next one (wrap around to start)
            j = (i + 1) % len(tour)
            dist += self.calculate_dist_matrix(self.cities[tour[i]], self.cities[tour[j]])
        return dist

    def two_opt_generator(self):
        """
        The core 2-opt heuristic implementation. 
        Yields the tour and distance whenever an improvement is found.
        """
        improved = True
        best_tour = self.tour.copy()
        num_cities = self.num_cities

        while improved:
            improved = False
            # O(N^2) complexity: iterate through all possible pairs of edges (i, i+1) and (j, j+1)
            for i in range(num_cities - 1):
                for j in range(i + 1, num_cities):
                    # Skip segments of length 1
                    if j - i <= 1:
                        continue
                    
                    # Perform the 2-opt segment reversal (the key swap operation)
                    new_tour = np.concatenate((
                        best_tour[:i+1],            # Part 1: Start to i (index i is included)
                        best_tour[i+1:j+1][::-1],   # Part 2: Segment i+1 to j is reversed
                        best_tour[j+1:]             # Part 3: j+1 to End
                    ))

                    new_dist = self.calculate_total_distance(new_tour)

                    if new_dist < self.current_dist:
                        self.current_dist = new_dist
                        best_tour = new_tour
                        improved = True
                        
                        # Yield the improved state for visualization
                        yield best_tour, self.current_dist
        
        # Yield the final optimized tour once the algorithm terminates
        yield best_tour, self.current_dist

    def update_plot(self, data):
        """
        Update function for Matplotlib animation, called for every yielded frame.
        
        Args:
            data (tuple): (current_tour, current_distance)
        """
        tour, dist = data
        
        # Prepare coordinates for plotting: append the first city to close the loop
        x = np.append(self.cities[tour, 0], self.cities[tour[0], 0])
        y = np.append(self.cities[tour, 1], self.cities[tour[0], 1])
        
        # Update the line data
        self.line.set_data(x, y)
        
        # Update the distance display text
        self.text.set_text(f'Total Distance: {dist:.2f}')
        
        return self.line, self.text

    def animate(self):
        """Setup and start the Matplotlib animation."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f"2-Opt Algorithm Visualization (N={self.num_cities})")
        
        # Set map boundaries
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal', adjustable='box') # Ensure squares look like squares
        
        # Plot cities as red dots
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=40, zorder=2)
        # Initialize the path line (blue). The comma is crucial for unpacking ax.plot output.
        self.line, = ax.plot([], [], 'b-', lw=1.5, zorder=1)
        # Initialize the distance display text
        self.text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Create animation using the generator as the frame source
        _ = animation.FuncAnimation(
            fig, 
            self.update_plot, 
            frames=self.optimizer, 
            interval=50, # Delay between frames in ms (adjust for speed)
            repeat=False, # Stop when generator is exhausted (algorithm is finished)
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    viz = TwoOptVisualizer(num_cities=40)
    viz.animate() 
