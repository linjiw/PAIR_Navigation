import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
def is_clear_for_robot(grid, x, y, robot_size=4):
    if x + robot_size > grid.shape[0] or y + robot_size > grid.shape[1]:
        return False
    for i in range(robot_size):
        for j in range(robot_size):
            if grid[x + i, y + j] != 0:
                return False
    return True

def find_path_for_robot(grid, robot_size=4):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque()
    predecessors = dict()

    for i in range(rows - robot_size + 1):
        if is_clear_for_robot(grid, i, 0, robot_size):
            queue.append((i, 0))
            visited[i, 0] = True

    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    path_found = False
    goal = None

    while queue:
        x, y = queue.popleft()

        if y + robot_size - 1 == cols - 1:
            goal = (x, y)
            path_found = True
            break

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < rows - robot_size + 1 and 0 <= new_y < cols - robot_size + 1 and not visited[new_x, new_y] and is_clear_for_robot(grid, new_x, new_y, robot_size):
                queue.append((new_x, new_y))
                visited[new_x, new_y] = True
                predecessors[(new_x, new_y)] = (x, y)

    if not path_found:
        return False, None  # No path found

    # Backtrack to find the path
    path = []
    while goal is not None:
        path.append(goal)
        goal = predecessors.get(goal)

    path.reverse()  # Reverse the path to start from the beginning

    return True, path

def draw_path(grid, path, robot_size=4):
    grid_with_path = np.copy(grid)
    for x, y in path:
        grid_with_path[x:x+robot_size, y:y+robot_size] = 2  # Mark the path
    
    # Create a color map for visualization
    cmap = colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(grid_with_path, cmap=cmap, norm=norm)
    plt.grid(which='both', color='lightgrey', linewidth=0.5)
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    plt.show()

def process_maps_in_folder(folder_path, robot_size=4):
    no_path_maps = []
    path_found_maps = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            grid = np.load(file_path)
            path_exists, path = find_path_for_robot(grid, robot_size)

            if path_exists:
                # print(f"Path found in {filename}")
                path_found_maps.append((grid, path))
            elif not path_exists:
                # print(f"No path found in {filename}")
                no_path_maps.append(grid)
    good_map_rate = len(path_found_maps) / (len(path_found_maps) + len(no_path_maps))
    print(f"good_map_rate: {good_map_rate}")
            # if len(path_found_maps) == 5 and len(no_path_maps) == 5:
            #     break
    return good_map_rate
    # for grid, path in path_found_maps:
    #     draw_path(grid, path, robot_size)

    # for grid in no_path_maps:
    #     draw_grid(grid)  # Assume draw_grid is a function similar to draw_path but only shows the grid

def draw_grid(grid):
    plt.imshow(grid, cmap='gray')
    plt.show()

# Replace 'your_folder_path_here' with the actual path to your folder containing .npy files
folder_path = 'worlds_train'
process_maps_in_folder(folder_path, robot_size=5)


# np.random.seed(42)  # For reproducibility
# grid_size = 30
# grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.7, 0.3])
# grid[:, :5] = 0  # Ensure start area is clear
# grid[:, -5:] = 0  # Ensure end area is clear
# # Example usage:
# path_exists, path = find_path_for_robot(grid, robot_size=4)
# if path_exists:
#     print("Path found!")
#     draw_path(grid, path, robot_size=4)
# else:
#     print("No path found.")
