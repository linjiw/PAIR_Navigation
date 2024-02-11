import os

# List of directories to create
directories = [
    './jackal_map_create/test_data/world_files',
    './jackal_map_create/test_data/grid_files',
    './jackal_map_create/test_data/cspace_files',
    './jackal_map_create/test_data/path_files',
    './jackal_map_create/test_data/metrics_files',
    './jackal_map_create/test_data/map_files'
]

# Create each directory
for directory in directories:
    os.makedirs(directory, exist_ok=True)
