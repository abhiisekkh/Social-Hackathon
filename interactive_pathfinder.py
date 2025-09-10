#!/usr/bin/env python3
"""
Interactive pathfinding visualization with refresh/restart controls.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import networkx as nx
from pathfinder import PathFinder
from collections import deque
from typing import List, Tuple, Set, Optional
import time

class InteractivePathFinder(PathFinder):
    def __init__(self, grid: List[List[int]], k: int):
        super().__init__(grid, k)
        self.fig = None
        self.ax = None
        self.ax_analysis = None
        self.ax_graph = None
        self.character_marker = None
        self.animation = None
        self.current_step = 0
        self.path = []
        self.broken_walls = set()
        self.trail_markers = []
        self.is_playing = False
        self.algorithm_stats = {}
        self.visited_states = {}
        self.exploration_order = []
        
    def create_interactive_visualization(self, path: List[Tuple[int, int]], broken_walls: Set[Tuple[int, int]]):
        """
        Create an interactive visualization with controls for restart, play/pause, and step control.
        """
        if not path:
            print("No path to visualize!")
            return
            
        self.path = path
        self.broken_walls = broken_walls
        self.current_step = 0
        
        # Set up the figure with proper spacing for fullscreen
        self.fig = plt.figure(figsize=(18, 10))
        
        # Main maze area (left side) - larger for better visibility
        self.ax = plt.subplot2grid((10, 18), (0, 0), colspan=8, rowspan=8)
        
        # Algorithm analysis area (top right) - more space
        self.ax_analysis = plt.subplot2grid((10, 18), (0, 9), colspan=9, rowspan=4)
        self.ax_analysis.axis('off')
        
        # DSA graph visualization (bottom right) - separate space
        self.ax_graph = plt.subplot2grid((10, 18), (4, 9), colspan=9, rowspan=4)
        
        # Control panel area (bottom) - fixed position
        self.control_ax = plt.subplot2grid((10, 18), (8, 0), colspan=18, rowspan=2)
        self.control_ax.set_xlim(0, 10)
        self.control_ax.set_ylim(0, 2)
        self.control_ax.axis('off')
        
        # Create the base maze visualization
        self._setup_maze_display()
        
        # Setup algorithm analysis
        self._setup_algorithm_analysis()
        
        # Setup DSA graph visualization
        self._setup_dsa_graph()
        
        # Create control buttons
        self._setup_controls()
        
        # Initialize character at start position
        self._update_display()
        
        # Use subplots_adjust instead of tight_layout for better control
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
        plt.show()
        
    def _setup_maze_display(self):
        """Set up the main maze display area."""
        # Create base colors
        colors = np.zeros((self.n, self.n, 3))
        
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) == (0, 0):
                    colors[i][j] = [0, 1, 0]  # Green for start
                elif (i, j) == (self.n-1, self.n-1):
                    colors[i][j] = [1, 0, 0]  # Red for destination
                elif (i, j) in self.broken_walls:
                    colors[i][j] = [1, 1, 0]  # Yellow for broken walls
                elif self.grid[i][j] == 1:
                    colors[i][j] = [0, 0, 0]  # Black for walls
                else:
                    colors[i][j] = [1, 1, 1]  # White for open cells
        
        self.ax.imshow(colors)
        
        # Add grid lines
        for i in range(self.n + 1):
            self.ax.axhline(i - 0.5, color='gray', linewidth=1)
            self.ax.axvline(i - 0.5, color='gray', linewidth=1)
        
        # Add crosses for broken walls
        for wall_row, wall_col in self.broken_walls:
            self.ax.plot([wall_col - 0.3, wall_col + 0.3], [wall_row - 0.3, wall_row + 0.3], 'r-', linewidth=3)
            self.ax.plot([wall_col - 0.3, wall_col + 0.3], [wall_row + 0.3, wall_row - 0.3], 'r-', linewidth=3)
        
        # Draw alternate paths with dotted lines
        if hasattr(self, 'algorithm_stats') and 'alternate_paths' in self.algorithm_stats:
            for i, alt_path in enumerate(self.algorithm_stats['alternate_paths']):
                path_coords = alt_path['path']
                if len(path_coords) > 1:
                    x_coords = [coord[1] for coord in path_coords]
                    y_coords = [coord[0] for coord in path_coords]
                    # Use different colors and dotted lines for alternate paths
                    colors_alt = ['purple', 'orange', 'brown', 'pink']
                    color = colors_alt[i % len(colors_alt)]
                    self.ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                               linestyle='--', alpha=0.7, zorder=5,
                               label=f'Alt Path {i+1} (L:{alt_path["length"]}, W:{alt_path["walls_broken"]})')
        
        # Draw optimal path with solid line (on top)
        if len(self.path) > 1:
            x_coords = [coord[1] for coord in self.path]
            y_coords = [coord[0] for coord in self.path]
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.8, zorder=6,
                        label=f'Optimal Path (L:{len(self.path)-1}, W:{len(self.broken_walls)})')
        
        # Add legend if there are multiple paths
        if hasattr(self, 'algorithm_stats') and self.algorithm_stats.get('total_paths_found', 0) > 1:
            self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
        
        # Initialize character marker
        self.character_marker = self.ax.plot([], [], 'o', markersize=20, color='orange', 
                                           markeredgecolor='black', markeredgewidth=2, zorder=10)[0]
        
        # Add character face
        self.character_face = self.ax.text(0, 0, '🚶', fontsize=24, ha='center', va='center', zorder=11)
        
        self.ax.set_xlim(-0.5, self.n - 0.5)
        self.ax.set_ylim(-0.5, self.n - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'🎮 Interactive Path Traversal (Length: {len(self.path) - 1})', fontsize=16, fontweight='bold')
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label='🏁 Start'),
            patches.Patch(color='red', label='🎯 Destination'),
            patches.Patch(color='white', label='⚪ Open Cell'),
            patches.Patch(color='black', label='⚫ Wall'),
            patches.Patch(color='yellow', label='💥 Broken Wall'),
            patches.Patch(color='orange', label='🚶 Character'),
            patches.Patch(color='lightblue', label='👣 Trail')
        ]
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def _setup_controls(self):
        """Set up interactive control buttons and sliders."""
        # Restart button - positioned for fullscreen visibility
        restart_ax = plt.axes([0.05, 0.05, 0.08, 0.06])
        self.restart_btn = Button(restart_ax, 'Restart', color='lightcoral', hovercolor='red')
        self.restart_btn.on_clicked(self._restart_animation)
        
        # Play/Pause button
        play_ax = plt.axes([0.15, 0.05, 0.08, 0.06])
        self.play_btn = Button(play_ax, 'Play', color='lightgreen', hovercolor='green')
        self.play_btn.on_clicked(self._toggle_play_pause)
        
        # Step forward button
        step_ax = plt.axes([0.25, 0.05, 0.08, 0.06])
        self.step_btn = Button(step_ax, 'Step', color='lightblue', hovercolor='blue')
        self.step_btn.on_clicked(self._step_forward)
        
        # Speed slider
        speed_ax = plt.axes([0.4, 0.05, 0.15, 0.06])
        self.speed_slider = Slider(speed_ax, 'Speed', 0.1, 3.0, valinit=1.0, valfmt='%.1fx')
        
        # Progress slider - positioned higher for visibility
        progress_ax = plt.axes([0.05, 0.12, 0.5, 0.04])
        self.progress_slider = Slider(progress_ax, 'Progress', 0, len(self.path)-1, 
                                    valinit=0, valfmt='Step %d', valstep=1)
        self.progress_slider.on_changed(self._on_progress_change)
        
        # Status text - positioned in control area
        self.status_text = self.control_ax.text(5, 1.5, f'Ready - Step 0/{len(self.path)}', 
                                              ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Instructions - smaller font to prevent collision
        self.control_ax.text(5, 0.3, 
                           'Controls: Restart | Play/Pause | Step Forward | Drag Progress Bar', 
                           ha='center', va='center', fontsize=9, style='italic')
    
    def _restart_animation(self, event):
        """Restart the animation from the beginning."""
        self.current_step = 0
        self.is_playing = False
        self.play_btn.label.set_text('Play')
        
        # Clear trail markers
        for marker in self.trail_markers:
            marker.remove()
        self.trail_markers.clear()
        
        # Reset progress slider
        self.progress_slider.reset()
        
        # Update display
        self._update_display()
        
        print("🔄 Animation restarted!")
    
    def _toggle_play_pause(self, event):
        """Toggle between play and pause."""
        if self.is_playing:
            self.is_playing = False
            self.play_btn.label.set_text('Play')
            if self.animation:
                self.animation.stop()
        else:
            self.is_playing = True
            self.play_btn.label.set_text('Pause')
            self._start_auto_play()
    
    def _step_forward(self, event):
        """Move one step forward in the animation."""
        if self.current_step < len(self.path) - 1:
            self.current_step += 1
            self.progress_slider.set_val(self.current_step)
            self._update_display()
    
    def _on_progress_change(self, val):
        """Handle progress slider changes."""
        new_step = int(val)
        if new_step != self.current_step:
            self.current_step = new_step
            self._update_display_from_slider()
    
    def _update_display(self):
        """Update the display based on current step."""
        if self.current_step >= len(self.path):
            return
            
        row, col = self.path[self.current_step]
        
        # Update character position
        self.character_marker.set_data([col], [row])
        self.character_face.set_position((col, row))
        
        # Add trail marker for previous position (except start)
        if self.current_step > 0:
            prev_row, prev_col = self.path[self.current_step - 1]
            if (prev_row, prev_col) != (0, 0):  # Don't trail over start
                trail = self.ax.plot(prev_col, prev_row, 's', markersize=8, 
                                   color='lightblue', alpha=0.7, zorder=5)[0]
                self.trail_markers.append(trail)
        
        # Update status
        self.status_text.set_text(f'Position ({row}, {col}) - Step {self.current_step + 1}/{len(self.path)}')
        
        # Check if we broke a wall at this position
        if (row, col) in self.broken_walls:
            self.status_text.set_text(f'💥 WALL BROKEN at ({row}, {col}) - Step {self.current_step + 1}/{len(self.path)}')
        
        # Update title
        self.ax.set_title(f'🎮 Interactive Path Traversal - Step {self.current_step + 1}/{len(self.path)}', 
                         fontsize=16, fontweight='bold')
        
        # Check if reached destination
        if self.current_step == len(self.path) - 1:
            self.status_text.set_text(f'🎉 DESTINATION REACHED! - Final Step {self.current_step + 1}/{len(self.path)}')
            self.is_playing = False
            self.play_btn.label.set_text('▶️ Play')
        
        self.fig.canvas.draw()
    
    def _update_display_from_slider(self):
        """Update display when slider is moved manually."""
        # Clear existing trail markers
        for marker in self.trail_markers:
            marker.remove()
        self.trail_markers.clear()
        
        # Rebuild trail up to current step
        for i in range(self.current_step):
            trail_row, trail_col = self.path[i]
            if (trail_row, trail_col) != (0, 0):  # Don't trail over start
                trail = self.ax.plot(trail_col, trail_row, 's', markersize=8, 
                                   color='lightblue', alpha=0.7, zorder=5)[0]
                self.trail_markers.append(trail)
        
        # Update character position
        if self.current_step < len(self.path):
            row, col = self.path[self.current_step]
            self.character_marker.set_data([col], [row])
            self.character_face.set_position((col, row))
            
            # Update status
            self.status_text.set_text(f'Position ({row}, {col}) - Step {self.current_step + 1}/{len(self.path)}')
            
            # Check if we broke a wall at this position
            if (row, col) in self.broken_walls:
                self.status_text.set_text(f'💥 WALL BROKEN at ({row}, {col}) - Step {self.current_step + 1}/{len(self.path)}')
        
        self.fig.canvas.draw()
    
    def _start_auto_play(self):
        """Start automatic playback."""
        def animate_step():
            if self.is_playing and self.current_step < len(self.path) - 1:
                self.current_step += 1
                self.progress_slider.set_val(self.current_step)
                self._update_display()
                
                # Schedule next step
                speed = self.speed_slider.val
                delay = int(1000 / speed)  # Convert speed to delay in ms
                self.fig.canvas.draw_idle()
                self.animation = self.fig.canvas.new_timer(interval=delay)
                self.animation.single_shot = True
                self.animation.add_callback(animate_step)
                self.animation.start()
            else:
                self.is_playing = False
                self.play_btn.label.set_text('Play')
        
        animate_step()
    
    def find_shortest_path_with_analysis(self) -> Tuple[int, List[Tuple[int, int]], Set[Tuple[int, int]], dict]:
        """Enhanced pathfinding with detailed algorithm analysis and multiple path detection."""
        start_time = time.time()
        
        # Check if we can start - if start is a wall, we need to break it
        start_walls_broken = 1 if self.grid[0][0] == 1 else 0
        start_broken_walls = {(0, 0)} if self.grid[0][0] == 1 else set()
        
        # If we can't even break the starting wall, return -1
        if start_walls_broken > self.k:
            return -1, [], set(), {}
        
        # Reset analysis data
        self.visited_states = {}
        self.exploration_order = []
        self.all_paths = []  # Store all valid paths found
        
        # Use priority queue to prioritize fewer walls broken first
        # Priority: (walls_broken, path_length, row, col, path, broken_walls)
        import heapq
        queue = [(start_walls_broken, 0, 0, 0, [(0, 0)], start_broken_walls)]
        heapq.heapify(queue)
        self.visited_states[(0, 0, start_walls_broken)] = 0
        states_explored = 0
        max_queue_size = 1
        optimal_length = float('inf')
        optimal_walls = float('inf')
        
        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            walls_broken, path_length, row, col, path, broken_walls = heapq.heappop(queue)
            states_explored += 1
            
            # Track exploration order for visualization
            self.exploration_order.append((row, col, walls_broken, len(path) - 1))
            
            # Check if we reached the destination
            if row == self.n - 1 and col == self.n - 1:
                actual_path_length = len(path) - 1
                walls_count = len(broken_walls)
                
                # Store this path
                self.all_paths.append({
                    'path': path.copy(),
                    'broken_walls': broken_walls.copy(),
                    'length': actual_path_length,
                    'walls_broken': walls_count,
                    'priority_score': walls_count * 1000 + actual_path_length  # Prioritize fewer walls, then shorter path
                })
                
                # Update optimal criteria
                if walls_count < optimal_walls or (walls_count == optimal_walls and actual_path_length < optimal_length):
                    optimal_walls = walls_count
                    optimal_length = actual_path_length
                
                # Continue searching for more paths (but limit to avoid infinite search)
                if len(self.all_paths) >= 5:  # Limit to 5 paths for performance
                    break
                continue
            
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
                    if state not in self.visited_states or self.visited_states[state] > current_steps:
                        self.visited_states[state] = current_steps
                        new_path = path + [(new_row, new_col)]
                        new_path_length = len(new_path) - 1
                        heapq.heappush(queue, (new_walls_broken, new_path_length, new_row, new_col, new_path, new_broken_walls))
        
        end_time = time.time()
        
        # Process found paths
        if self.all_paths:
            # Sort paths by priority (fewer walls first, then shorter path)
            self.all_paths.sort(key=lambda x: x['priority_score'])
            optimal_path = self.all_paths[0]
            
            # Calculate algorithm statistics
            self.algorithm_stats = {
                'execution_time': end_time - start_time,
                'states_explored': states_explored,
                'max_queue_size': max_queue_size,
                'total_possible_states': self.n * self.n * (self.k + 1),
                'state_space_efficiency': states_explored / (self.n * self.n * (self.k + 1)),
                'path_length': optimal_path['length'],
                'walls_broken': optimal_path['walls_broken'],
                'theoretical_time_complexity': f"O(N² × K) = O({self.n}² × {self.k}) = O({self.n * self.n * self.k})",
                'theoretical_space_complexity': f"O(N² × K) = O({self.n}² × {self.k}) = O({self.n * self.n * self.k})",
                'actual_states_visited': len(self.visited_states),
                'is_optimal': True,
                'algorithm': 'BFS with 3D State Space',
                'total_paths_found': len(self.all_paths),
                'alternate_paths': self.all_paths[1:] if len(self.all_paths) > 1 else []
            }
            
            return optimal_path['length'], optimal_path['path'], optimal_path['broken_walls'], self.algorithm_stats
        
        # No path found statistics
        self.algorithm_stats = {
            'execution_time': end_time - start_time,
            'states_explored': states_explored,
            'max_queue_size': max_queue_size,
            'total_possible_states': self.n * self.n * (self.k + 1),
            'state_space_efficiency': states_explored / (self.n * self.n * (self.k + 1)),
            'path_length': -1,
            'walls_broken': 0,
            'theoretical_time_complexity': f"O(N² × K) = O({self.n}² × {self.k}) = O({self.n * self.n * self.k})",
            'theoretical_space_complexity': f"O(N² × K) = O({self.n}² × {self.k}) = O({self.n * self.n * self.k})",
            'actual_states_visited': len(self.visited_states),
            'is_optimal': True,
            'algorithm': 'BFS with 3D State Space',
            'total_paths_found': 0,
            'alternate_paths': []
        }
        
        return -1, [], set(), self.algorithm_stats
    
    def _setup_algorithm_analysis(self):
        """Setup the algorithm analysis display."""
        self.ax_analysis.clear()
        self.ax_analysis.axis('off')
        
        # Title
        self.ax_analysis.text(0.5, 0.95, '📊 Algorithm Analysis', ha='center', va='top', 
                             fontsize=14, fontweight='bold', transform=self.ax_analysis.transAxes)
        
        # Algorithm info - adjusted spacing and font size
        y_pos = 0.9
        line_height = 0.09
        
        # Get path information
        total_paths = getattr(self.algorithm_stats, 'total_paths_found', 0) if hasattr(self, 'algorithm_stats') else 0
        alt_paths = getattr(self.algorithm_stats, 'alternate_paths', []) if hasattr(self, 'algorithm_stats') else []
        
        analysis_text = [
            f"Algorithm: BFS with 3D State Space",
            f"Time Complexity: O(N² × K) = O({self.n}² × {self.k})",
            f"Space Complexity: O(N² × K) = O({self.n}² × {self.k})",
            f"Optimality: Guaranteed Optimal",
            "",
            "Path Analysis:",
            f"  • Total Paths Found: {total_paths}",
            f"  • Optimal Path Length: {len(self.path)-1 if self.path else 'N/A'}",
            f"  • Optimal Walls Broken: {len(self.broken_walls)}/{self.k}",
        ]
        
        # Add alternate path information
        if alt_paths:
            analysis_text.append("  • Alternate Paths:")
            for i, alt_path in enumerate(alt_paths[:3]):  # Show max 3 alternates
                analysis_text.append(f"    Path {i+2}: L={alt_path['length']}, W={alt_path['walls_broken']}")
        
        analysis_text.extend([
            "",
            "Why BFS is Optimal:",
            "  • Explores states level by level",
            "  • Prioritizes fewer walls broken",
            "  • Then prioritizes shorter path length",
            "  • State (r,c,k) = min steps to reach (r,c)",
            "",
            "Current Grid Analysis:",
            f"  • Grid Size: {self.n}×{self.n} = {self.n*self.n} cells",
            f"  • Max Wall Breaks: {self.k}",
            f"  • Total States: {self.n*self.n*(self.k+1):,}"
        ])
        
        for i, text in enumerate(analysis_text):
            self.ax_analysis.text(0.02, y_pos - i * line_height, text, ha='left', va='top', 
                                 fontsize=8, transform=self.ax_analysis.transAxes,
                                 fontweight='bold' if text.startswith(('Why BFS', 'Current Grid')) else 'normal')
    
    def _setup_dsa_graph(self):
        """Setup the DSA graph visualization."""
        self.ax_graph.clear()
        
        # Create a simplified graph showing BFS exploration
        G = nx.Graph()
        
        if not self.path:
            self.ax_graph.text(0.5, 0.5, 'No Path to Visualize', ha='center', va='center', 
                              fontsize=12, transform=self.ax_graph.transAxes)
            return
        
        # Add nodes for each position in the path
        pos_dict = {}
        node_colors = []
        node_labels = {}
        
        for i, (row, col) in enumerate(self.path):
            node_id = f"({row},{col})".replace(' ', '')
            G.add_node(node_id)
            
            # Position nodes in a layout that shows progression
            pos_dict[node_id] = (i * 0.8, -row * 0.3 + col * 0.2)
            node_labels[node_id] = f"{row},{col}"
            
            # Color nodes based on type
            if (row, col) == (0, 0):
                node_colors.append('green')
            elif (row, col) == (self.n-1, self.n-1):
                node_colors.append('red')
            elif (row, col) in self.broken_walls:
                node_colors.append('yellow')
            else:
                node_colors.append('lightblue')
        
        # Add edges between consecutive path nodes
        for i in range(len(self.path) - 1):
            node1 = f"({self.path[i][0]},{self.path[i][1]})".replace(' ', '')
            node2 = f"({self.path[i+1][0]},{self.path[i+1][1]})".replace(' ', '')
            G.add_edge(node1, node2)
        
        # Draw the graph with better spacing
        if G.nodes():
            nx.draw(G, pos_dict, ax=self.ax_graph, node_color=node_colors, 
                   node_size=150, font_size=6, font_weight='bold',
                   edge_color='gray', width=1.2, with_labels=True, labels=node_labels)
        
        self.ax_graph.set_title('BFS State Space Graph\n(Path Progression)', 
                               fontsize=10, fontweight='bold')
        
        # Add legend with smaller font
        legend_elements = [
            patches.Patch(color='green', label='Start'),
            patches.Patch(color='red', label='Goal'),
            patches.Patch(color='yellow', label='Wall Broken'),
            patches.Patch(color='lightblue', label='Path Node')
        ]
        self.ax_graph.legend(handles=legend_elements, loc='upper right', fontsize=7)

def get_user_input():
    """Get user input for grid and parameters according to requirements."""
    print("🎮 INTERACTIVE PATHFINDING DEMO")
    print("=" * 50)
    
    # Get K (maximum walls to break) - single input line
    k = int(input("K (max walls to break): "))
    
    # Choose input method
    print("\nGrid Input Options:")
    print("1. Read from CSV file")
    print("2. Generate random grid")
    
    choice = input("Choice (1 or 2): ").strip()
    
    if choice == '1':
        # CSV file input
        filename = input("CSV filename: ").strip()
        try:
            grid = PathFinder.read_grid_from_csv(filename)
            print(f"Loaded {len(grid)}x{len(grid[0])} grid from {filename}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            print("Using default 5x5 grid instead")
            grid = [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0]
            ]
    else:
        # Random generation
        n = int(input("Grid size N (for NxN): "))
        wall_prob = float(input("Wall probability (0.0-1.0): "))
        grid = PathFinder.generate_random_grid(n, wall_prob)
        print(f"Generated {n}x{n} random grid")
    
    return grid, k

def demo_interactive_pathfinding():
    """Demonstrate the interactive pathfinding visualization."""
    # Get user input
    grid, k = get_user_input()
    
    # Create interactive pathfinder
    interactive_pf = InteractivePathFinder(grid, k)
    
    # Find path with analysis
    length, path, broken_walls, stats = interactive_pf.find_shortest_path_with_analysis()
    
    # Output 1: Single integer shortest path length
    print(f"\nOutput 1: {length}")
    
    if length == -1:
        print("No path possible")
        return
    
    # Output 2: Save CSV file with path marked
    output_csv = "shortest_path_output.csv"
    interactive_pf.save_path_to_csv(path, broken_walls, output_csv)
    print(f"Output 2: Path saved to {output_csv}")
    
    print(f"\nPath details:")
    print(f"Length: {length}")
    print(f"Walls broken: {len(broken_walls)}")
    print(f"Execution time: {stats['execution_time']:.4f}s")
    print(f"Algorithm: {stats['algorithm']}")
    print(f"Time complexity: {stats['theoretical_time_complexity']}")
    
    # Create interactive visualization
    interactive_pf.create_interactive_visualization(path, broken_walls)

if __name__ == "__main__":
    demo_interactive_pathfinding()
