import numpy as np
import math
import queue
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridMetricsAnalyzer:
    def __init__(self, grids, disp_radius=3):
        self.grids = grids
        self.disp_radius = disp_radius
    def analyze(self):
        # Aggregated lists to hold metrics for all grids
        aggregated_cw = []
        aggregated_av = []
        aggregated_dp = []
        aggregated_cd = []
        aggregated_or = []
        # Calculate and aggregate metrics for each grid
        for grid in self.grids:
            cw = self.closest_wall(grid)
            av = self.avg_visibility(grid)
            dp = self.dispersion(grid)
            cd = self.characteristic_dimension(grid)
            or_value = self.occupancy_rate(grid)
            aggregated_cw.extend(cw)
            aggregated_av.extend(av)
            aggregated_dp.extend(dp)
            aggregated_cd.extend(cd)
            aggregated_or.append(or_value)

        # Calculate mean, std, min, max for each metric
        results = {
            'closest_wall': self._calculate_stats(aggregated_cw),
            'avg_visibility': self._calculate_stats(aggregated_av),
            'dispersion': self._calculate_stats(aggregated_dp),
            'characteristic_dimension': self._calculate_stats(aggregated_cd),
            'occupancy_rate': self._calculate_stats(aggregated_or)

        }

        return results

    def occupancy_rate(self, grid):
        """
        Calculate the occupancy rate of the grid.
        Occupied cells are assumed to be represented by 1.
        """
        rows, cols = len(grid), len(grid[0])
        occupied_count = sum(sum(1 for cell in row if cell == 1) for row in grid)
        total_cells = rows * cols
        return occupied_count / total_cells if total_cells > 0 else 0


    def _calculate_stats(self, metric_list):
        metric_array = np.array(metric_list)
        return {
            'mean': np.mean(metric_array),
            'std': np.std(metric_array),
            'min': np.min(metric_array),
            'max': np.max(metric_array)
        }
    def closest_wall(self, grid):
        rows, cols = len(grid), len(grid[0])
        dists = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                dists[r][c] = self._bfs_distance(grid, r, c)
        return dists
    def print_results(self, results):
        for metric, stats in results.items():
            print(f"{metric.title()} Metric:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Standard Deviation: {stats['std']:.2f}")
            print(f"  Minimum: {stats['min']:.2f}")
            print(f"  Maximum: {stats['max']:.2f}\n")
    def avg_visibility(self, grid):
        directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        rows, cols = len(grid), len(grid[0])
        vis = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                vis[r][c] = self._avg_visibility_from_cell(grid, r, c, directions)
        return vis

    def dispersion(self, grid):
        directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        rows, cols = len(grid), len(grid[0])
        disp = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                disp[r][c] = self._cell_dispersion(grid, r, c, directions)
        return disp

    def characteristic_dimension(self, grid):
        directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        rows, cols = len(grid), len(grid[0])
        cdr = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                cdr[r][c] = self._characteristic_dimension_from_cell(grid, r, c, directions)
        return cdr

    def _bfs_distance(self, grid, start_row, start_col):
        if grid[start_row][start_col] == 1:
            return 0  # Starting point is an obstacle

        rows, cols = len(grid), len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        q = queue.Queue()
        q.put((start_row, start_col, 0))  # (row, col, distance)

        while not q.empty():
            r, c, dist = q.get()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 directions
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    if grid[nr][nc] == 1:
                        return dist + 1
                    q.put((nr, nc, dist + 1))
                    visited[nr][nc] = True
        return rows + cols  # Max distance if no obstacle found

    def _avg_visibility_from_cell(self, grid, r, c, directions):
        visibility_sum = 0
        for dr, dc in directions:
            step = 1
            while self._in_grid(grid, r + step*dr, c + step*dc) and grid[r + step*dr][c + step*dc] == 0:
                step += 1
            visibility_sum += step
        return visibility_sum / len(directions)

    def _cell_dispersion(self, grid, r, c, directions):
        change_count = 0
        for dr, dc in directions:
            if not self._in_grid(grid, r + dr, c + dc):
                continue
            if grid[r][c] != grid[r + dr][c + dc]:
                change_count += 1
        return change_count

    def _characteristic_dimension_from_cell(self, grid, r, c, directions):
        min_visibility = float('inf')
        for dr, dc in directions:
            visibility = 0
            step = 1
            while self._in_grid(grid, r + step*dr, c + step*dc) and grid[r + step*dr][c + step*dc] == 0:
                step += 1
                visibility = step
            min_visibility = min(min_visibility, visibility)
        return min_visibility

    def _in_grid(self, grid, r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[0])
    
import os
def load_grids_from_folder(folder_path):
    grids = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            grid_path = os.path.join(folder_path, file)
            grid = np.load(grid_path)
            grids.append(grid)
    return grids

def test_grid_metrics(folder_path):
    grids = load_grids_from_folder(folder_path)
    if not grids:
        print("No grid maps found in the folder.")
        return

    analyzer = GridMetricsAnalyzer(grids)
    results = analyzer.analyze()
    # analyzer.print_results(results)
    return results
    



# Function to plot metrics
def plot_metric(all_results, metric_name):
    means = [result[metric_name]['mean'] for result in all_results]
    stds = [result[metric_name]['std'] for result in all_results]
    mins = [result[metric_name]['min'] for result in all_results]
    maxs = [result[metric_name]['max'] for result in all_results]
    folders = range(1, len(all_results) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(folders, means, marker='o', label='Mean')
    plt.plot(folders, stds, marker='s', label='Standard Deviation')
    plt.plot(folders, mins, marker='^', label='Minimum')
    plt.plot(folders, maxs, marker='v', label='Maximum')
    plt.xlabel('Map Folder')
    plt.ylabel(metric_name.title())
    plt.title(f'{metric_name.title()} Across Different Map Folders')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name}_metrics_plot.png')
    plt.show()


# # Collect results for all folders
# all_results = []
# for i in range(100):
#     if i != 1:
#         folder_path = f'/home/linjiw/Downloads/PAIRED-Jackal/assets/urdf/jackal/worlds{i}'
#         result = test_grid_metrics(folder_path)
#         if result:
#             all_results.append(result)

# # Plot and save metrics
# metrics = ['closest_wall', 'avg_visibility', 'dispersion', 'characteristic_dimension']
# for metric in metrics:
#     plot_metric(all_results, metric)


def test_metrics_for_each_map(folder_path):
    grids = load_grids_from_folder(folder_path)
    if not grids:
        print("No grid maps found in the folder.")
        return None

    results = {'closest_wall': [], 'avg_visibility': [], 'dispersion': [], 'characteristic_dimension': [], 'occupancy_rate': []}
    
    for grid in tqdm(grids, desc="Analyzing maps"):
        analyzer = GridMetricsAnalyzer([grid])
        grid_result = analyzer.analyze()
        for metric in results.keys():
            results[metric].append(grid_result[metric]['mean'])  # Since each grid has only one value
    # print(results)
    
    
    # Calculate mean for each metric
    mean_results = {metric: sum(values) / len(values) for metric, values in results.items() if values}
    
    # print("Individual Results:", results)
    print("Mean Results:", mean_results)
    return mean_results
    # return results


def plot_metrics_for_each_map(results, folder_path):
    map_indices = range(len(results['closest_wall']))
    for metric in results.keys():
        plt.figure(figsize=(12, 6))
        plt.plot(map_indices, results[metric], marker='o', label=metric.title())
        plt.xlabel('Map Index')
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} for Each Map in {folder_path}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{folder_path.replace("/", "_")}_{metric}_plot.png')
        plt.show()


if __name__ == '__main__':
    folder_path = '/home/linjiw/Downloads/PAIRED-Jackal/assets/urdf/jackal/worlds2'
    # folder_path = "/home/linjiw/Downloads/jackal-map-creation/test_data/grid_files"
    folder_results = test_metrics_for_each_map(folder_path)
    # if folder_results:
    #     plot_metrics_for_each_map(folder_results, folder_path)
# Replace 'path_to_single_folder' with the actual path of the folder
# folder_path = '/home/linjiw/Downloads/PAIRED-Jackal/assets/urdf/jackal/worlds'
# # folder_path = "/home/linjiw/Downloads/jackal-map-creation/test_data/grid_files"
# folder_results = test_metrics_for_each_map(folder_path)
# # if folder_results:
#     plot_metrics_for_each_map(folder_results, folder_path)

# random.seed(0)  # Seed for reproducibility
# test_grids = [np.random.choice([0, 1], size=(30, 30), p=[0.6, 0.4]) for _ in range(10)]  # 10 test grids

# # Testing the GridMetricsAnalyzer
# analyzer = GridMetricsAnalyzer(test_grids)
# results = analyzer.analyze()

# # Print the results
# analyzer.print_results(results)