# Shortest Path Finder with Wall Breaking

A Python solution for finding the shortest path in an N×N grid where you can break at most K walls to reach from (0,0) to (N-1,N-1).

## Problem Description

Given an N×N grid where:
- `0` represents an open cell
- `1` represents a wall
- You start at position (0,0)
- You want to reach position (N-1,N-1)
- You may break at most K walls during your journey

Find the length of the shortest path, or return -1 if no path is possible.

## Features

### Core Algorithm
- **BFS with State Tracking**: Uses Breadth-First Search with state (row, col, walls_broken) to guarantee shortest path
- **Optimal Solution**: Finds the true shortest path considering wall breaking constraints
- **Efficient Implementation**: Avoids redundant state exploration

### Input Options
1. **CSV File Input**: Read grid from existing CSV file
2. **Random Generation**: Generate random N×N grid with configurable wall probability

### Output Formats
1. **Path Length**: Single integer representing shortest path length (-1 if impossible)
2. **CSV Output**: Grid with path marked using special characters:
   - `S`: Start position (0,0)
   - `E`: End position (N-1,N-1)
   - `x`: Path cells
   - `B`: Broken walls
   - `0`/`1`: Original open cells/walls

### Visualization
- **Color-coded Grid**:
  - White: Open cells
  - Black: Walls
  - Green: Start position
  - Red: Destination
  - Light Blue: Path taken
  - Yellow: Broken walls (with red cross marks)
- **Interactive Display**: Shows grid with legend and path information
- **Save Options**: Export visualization as PNG image

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Program
```bash
python main.py
```

The program will guide you through:
1. Setting the maximum number of walls to break (K)
2. Choosing input method (CSV file or random generation)
3. Configuring output options
4. Viewing results and visualization

### Example CSV Input Format
```csv
0,0,1,0
1,0,1,0
0,0,0,0
0,1,1,0
```

### Programmatic Usage
```python
from pathfinder import PathFinder

# Create or load grid
grid = [[0,1,0], [0,1,0], [0,0,0]]
k = 1  # Can break 1 wall

# Find path
pathfinder = PathFinder(grid, k)
length, path, broken_walls = pathfinder.find_shortest_path()

print(f"Shortest path length: {length}")
print(f"Path coordinates: {path}")
print(f"Broken walls: {broken_walls}")

# Save results
pathfinder.save_path_to_csv(path, broken_walls, "output.csv")
pathfinder.visualize_path(path, broken_walls, "visualization.png")
```

## Algorithm Details

### BFS with State Tracking
The algorithm uses BFS where each state is represented as `(row, col, walls_broken)`. This ensures:
- **Shortest Path**: BFS guarantees the first path found is shortest
- **Wall Breaking Optimization**: Tracks minimum walls broken to reach each position
- **State Pruning**: Avoids exploring states that can't lead to better solutions

### Time Complexity
- **Time**: O(N² × K) where N is grid size and K is max walls to break
- **Space**: O(N² × K) for the visited states tracking

### Key Features
- Handles edge cases (blocked start/end positions)
- Efficient state management to avoid redundant exploration
- Comprehensive path reconstruction with broken wall tracking

## File Structure
```
├── main.py           # Main program with user interface
├── pathfinder.py     # Core PathFinder class and algorithms
├── requirements.txt  # Python dependencies
└── README.md        # This documentation
```

## Example Output

### Console Output
```
Shortest path length: 8
Number of walls broken: 2
Walls broken at positions: [(1, 1), (2, 3)]
Path saved to path_output.csv
Visualization saved to path_visualization.png
```

### CSV Output (path_output.csv)
```
S,x,1,0
1,x,1,x
0,x,x,x
0,1,1,E
```

## Dependencies
- `matplotlib>=3.5.0`: For visualization
- `numpy>=1.21.0`: For array operations
- Standard library: `csv`, `collections`, `random`, `typing`

## License
This project is provided as-is for educational and practical use.
