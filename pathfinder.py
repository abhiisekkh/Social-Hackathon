import csv
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Set
import time

class PathFinder:
    def __init__(self, grid: List[List[int]], k: int):
        """
        Initialize the PathFinder with a grid and maximum walls that can be broken.
        
        Args:
            grid: N x N grid where 0 = open cell, 1 = wall
            k: Maximum number of walls that can be broken
        """
        self.grid = grid
        self.n = len(grid)
        self.k = k
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
    def find_shortest_path(self) -> Tuple[int, List[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Find the shortest path from (0,0) to (N-1, N-1) using BFS with state tracking.
        
        ALGORITHM: BFS with 3D State Space (row, col, walls_broken)
        TIME COMPLEXITY: O(N² × K) where N = grid size, K = max walls to break
        SPACE COMPLEXITY: O(N² × K) for visited states storage
        
        WHY OPTIMAL: BFS guarantees shortest path in unweighted graphs.
        Each state (r,c,k) represents minimum steps to reach (r,c) with k walls broken.
        
        Returns:
            Tuple of (path_length, path_coordinates, broken_walls)
            Returns (-1, [], set()) if no path exists
        """
        # Check if we can start - if start is a wall, we need to break it
        start_walls_broken = 1 if self.grid[0][0] == 1 else 0
        start_broken_walls = {(0, 0)} if self.grid[0][0] == 1 else set()
        
        # If we can't even break the starting wall, return -1
        if start_walls_broken > self.k:
            return -1, [], set()
        
        # State: (row, col, walls_broken, path, broken_walls_set)
        queue = deque([(0, 0, start_walls_broken, [(0, 0)], start_broken_walls)])
        # Visited: (row, col, walls_broken) -> minimum steps to reach this state
        visited = {}
        visited[(0, 0, start_walls_broken)] = 0
        
        while queue:
            row, col, walls_broken, path, broken_walls = queue.popleft()
            
            # Check if we reached the destination
            if row == self.n - 1 and col == self.n - 1:
                return len(path) - 1, path, broken_walls
            
            # Explore all four directions
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < self.n and 0 <= new_col < self.n:
                    new_walls_broken = walls_broken
                    new_broken_walls = broken_walls.copy()
                    
                    # If it's a wall, we need to break it
                    if self.grid[new_row][new_col] == 1:
                        if walls_broken >= self.k:
                            continue  # Can't break more walls
                        new_walls_broken += 1
                        new_broken_walls.add((new_row, new_col))
                    
                    state = (new_row, new_col, new_walls_broken)
                    current_steps = len(path)
                    
                    # Only proceed if we haven't visited this state or found a better path
                    if state not in visited or visited[state] > current_steps:
                        visited[state] = current_steps
                        new_path = path + [(new_row, new_col)]
                        queue.append((new_row, new_col, new_walls_broken, new_path, new_broken_walls))
        
        return -1, [], set()
    
    @staticmethod
    def read_grid_from_csv(filename: str) -> List[List[int]]:
        """Read grid from CSV file."""
        grid = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                grid.append([int(cell) for cell in row])
        return grid
    
    @staticmethod
    def generate_random_grid(n: int, wall_probability: float = 0.3) -> List[List[int]]:
        """
        Generate a random N x N grid.
        
        Args:
            n: Size of the grid
            wall_probability: Probability of a cell being a wall
        """
        grid = [[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                # Ensure start and end positions are always open
                if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                    grid[i][j] = 0
                else:
                    grid[i][j] = 1 if random.random() < wall_probability else 0
        
        return grid
    
    def save_path_to_csv(self, path: List[Tuple[int, int]], broken_walls: Set[Tuple[int, int]], filename: str):
        """Save the grid with path marked as 'x' to CSV file."""
        # Create a copy of the grid with string representations
        result_grid = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if (i, j) in path:
                    if (i, j) == (0, 0):
                        row.append('S')  # Start
                    elif (i, j) == (self.n-1, self.n-1):
                        row.append('E')  # End
                    else:
                        row.append('x')  # Path
                elif (i, j) in broken_walls:
                    row.append('B')  # Broken wall
                else:
                    row.append(str(self.grid[i][j]))
            result_grid.append(row)
        
        # Write to CSV
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_grid)
    
    def visualize_path(self, path: List[Tuple[int, int]], broken_walls: Set[Tuple[int, int]], 
                      save_filename: Optional[str] = None):
        """
        Visualize the grid with the path highlighted.
        
        Args:
            path: List of coordinates representing the path
            broken_walls: Set of coordinates where walls were broken
            save_filename: Optional filename to save the visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Create color map
        colors = np.zeros((self.n, self.n, 3))
        
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) == (0, 0):
                    colors[i][j] = [0, 1, 0]  # Green for start
                elif (i, j) == (self.n-1, self.n-1):
                    colors[i][j] = [1, 0, 0]  # Red for destination
                elif (i, j) in broken_walls:
                    colors[i][j] = [0.8, 0.8, 0]  # Yellow for broken walls
                elif (i, j) in path:
                    colors[i][j] = [0, 0.7, 1]  # Light blue for path
                elif self.grid[i][j] == 1:
                    colors[i][j] = [0, 0, 0]  # Black for walls
                else:
                    colors[i][j] = [1, 1, 1]  # White for open cells
        
        ax.imshow(colors)
        
        # Add grid lines
        for i in range(self.n + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5)
        
        # Add crosses for broken walls
        for wall_row, wall_col in broken_walls:
            ax.plot([wall_col - 0.3, wall_col + 0.3], [wall_row - 0.3, wall_row + 0.3], 'r-', linewidth=3)
            ax.plot([wall_col - 0.3, wall_col + 0.3], [wall_row + 0.3, wall_row - 0.3], 'r-', linewidth=3)
        
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Shortest Path Visualization (Path Length: {len(path) - 1 if path else "No Path"})')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label='Start (0,0)'),
            patches.Patch(color='red', label='End (N-1,N-1)'),
            patches.Patch(color='white', label='Open Cell'),
            patches.Patch(color='black', label='Wall'),
            patches.Patch(color='lightblue', label='Path'),
            patches.Patch(color='yellow', label='Broken Wall')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        
        plt.show()
