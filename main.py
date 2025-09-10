#!/usr/bin/env python3
"""
Shortest Path Finder with Wall Breaking
Solves the problem of finding shortest path in an N x N grid where you can break at most K walls.
"""

import os
import sys
from pathfinder import PathFinder

def print_usage():
    """Print usage instructions."""
    print("Shortest Path Finder with Wall Breaking")
    print("=" * 50)
    print("Usage: python main.py")
    print("\nThis program finds the shortest path from (0,0) to (N-1,N-1)")
    print("in an N x N grid where you can break at most K walls.")
    print("\nInput options:")
    print("1. Read grid from CSV file")
    print("2. Generate random grid")
    print("\nOutput:")
    print("- Shortest path length (or -1 if impossible)")
    print("- CSV file with path marked")
    print("- Visual representation")

def get_user_input():
    """Get user input for the problem parameters."""
    print_usage()
    print("\n" + "=" * 50)
    
    # Get K (maximum walls to break)
    while True:
        try:
            k = int(input("Enter maximum number of walls you can break (K): "))
            if k >= 0:
                break
            else:
                print("K must be non-negative. Please try again.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Choose input method
    print("\nChoose input method:")
    print("1. Read grid from CSV file")
    print("2. Generate random grid")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2.")
    
    grid = None
    
    if choice == '1':
        # Read from CSV file
        while True:
            filename = input("Enter CSV filename (or path): ").strip()
            if os.path.exists(filename):
                try:
                    grid = PathFinder.read_grid_from_csv(filename)
                    print(f"Successfully loaded {len(grid)}x{len(grid[0])} grid from {filename}")
                    break
                except Exception as e:
                    print(f"Error reading file: {e}")
                    print("Please try again.")
            else:
                print(f"File '{filename}' not found. Please try again.")
    
    else:
        # Generate random grid
        while True:
            try:
                n = int(input("Enter grid size (N for N x N grid): "))
                if n > 0:
                    break
                else:
                    print("Grid size must be positive.")
            except ValueError:
                print("Please enter a valid integer.")
        
        while True:
            try:
                wall_prob = float(input("Enter wall probability (0.0 to 1.0, recommended: 0.3): "))
                if 0.0 <= wall_prob <= 1.0:
                    break
                else:
                    print("Wall probability must be between 0.0 and 1.0.")
            except ValueError:
                print("Please enter a valid number.")
        
        grid = PathFinder.generate_random_grid(n, wall_prob)
        print(f"Generated {n}x{n} random grid with wall probability {wall_prob}")
        
        # Save the generated grid
        save_grid = input("Save generated grid to CSV? (y/n): ").strip().lower()
        if save_grid == 'y':
            grid_filename = input("Enter filename to save grid (e.g., 'grid.csv'): ").strip()
            try:
                import csv
                with open(grid_filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(grid)
                print(f"Grid saved to {grid_filename}")
            except Exception as e:
                print(f"Error saving grid: {e}")
    
    return k, grid

def display_grid_info(grid):
    """Display basic information about the grid."""
    n = len(grid)
    total_cells = n * n
    wall_count = sum(sum(row) for row in grid)
    open_count = total_cells - wall_count
    
    print(f"\nGrid Information:")
    print(f"Size: {n} x {n}")
    print(f"Total cells: {total_cells}")
    print(f"Open cells: {open_count}")
    print(f"Walls: {wall_count}")
    print(f"Wall percentage: {wall_count/total_cells*100:.1f}%")
    
    # Check if start and end are accessible
    start_blocked = grid[0][0] == 1
    end_blocked = grid[n-1][n-1] == 1
    
    if start_blocked or end_blocked:
        print("WARNING: Start or end position is blocked!")
        if start_blocked:
            print("  - Start position (0,0) is blocked")
        if end_blocked:
            print(f"  - End position ({n-1},{n-1}) is blocked")

def main():
    """Main function to run the pathfinding program."""
    try:
        # Get user input
        k, grid = get_user_input()
        
        # Display grid information
        display_grid_info(grid)
        
        # Create PathFinder instance
        pathfinder = PathFinder(grid, k)
        
        print(f"\nFinding shortest path with at most {k} wall breaks...")
        
        # Find the shortest path
        path_length, path, broken_walls = pathfinder.find_shortest_path()
        
        # Display results
        print("\n" + "=" * 50)
        print("RESULTS:")
        print("=" * 50)
        
        if path_length == -1:
            print("No path found! It's impossible to reach the destination.")
            print("Try increasing K (number of walls you can break).")
        else:
            print(f"Shortest path length: {path_length}")
            print(f"Number of walls broken: {len(broken_walls)}")
            
            if broken_walls:
                print("Walls broken at positions:", sorted(list(broken_walls)))
            
            # Save path to CSV
            output_filename = input("\nEnter filename to save path CSV (e.g., 'path_output.csv'): ").strip()
            if not output_filename:
                output_filename = "path_output.csv"
            
            pathfinder.save_path_to_csv(path, broken_walls, output_filename)
            print(f"Path saved to {output_filename}")
            
            # Show visualization
            show_viz = input("Show visualization? (y/n): ").strip().lower()
            if show_viz == 'y':
                save_viz = input("Save visualization image? (y/n): ").strip().lower()
                viz_filename = None
                if save_viz == 'y':
                    viz_filename = input("Enter filename for visualization (e.g., 'path_viz.png'): ").strip()
                    if not viz_filename:
                        viz_filename = "path_visualization.png"
                
                print("Generating visualization...")
                pathfinder.visualize_path(path, broken_walls, viz_filename)
                
                if viz_filename:
                    print(f"Visualization saved to {viz_filename}")
        
        print("\nProgram completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
