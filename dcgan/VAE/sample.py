

import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from model import VAE
from dcgan.VAE.model import VAE  # Replace 'vae_model' with the actual module name
# from dataset import NumpyDataset  # Replace 'dataset' with the actual module name
import torch
import random


import os
import numpy as np
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

def load_model(model_path, latent_dim, device):
    # Initialize the model
    model = VAE(latent_dim=latent_dim).to(device)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def generate_images(model, num_samples, latent_dim, device, save_dir='generated_images', show_images=True):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Sample random vectors from the standard normal distribution
    random_latent_vectors = torch.randn(num_samples, latent_dim).to(device)

    # Generate images from the random latent vectors
    with torch.no_grad():
        generated_images = model.decoder(random_latent_vectors)

    # Reshape the output to image format
    generated_images = generated_images.view(-1, 1, 30, 30)

    for i, image in enumerate(generated_images):
        # Save each image
        save_path = os.path.join(save_dir, f"generated_image_{i}.png")
        save_image(image, save_path)

        # Visualize each image
        if show_images:
            plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Generated Image {i}")
            plt.show()

    return generated_images

def load_npy_files_from_folder(folder_path):
    npy_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            npy_data = np.load(file_path)
            npy_files.append(npy_data)
    return npy_files

# Replace 'your_folder_path_here' with the actual path to your folder containing .npy files
folder_path = '/home/linjiw/Downloads/PAIR_Navigation/assets/urdf/jackal/worlds_eval'
npy_files_list = load_npy_files_from_folder(folder_path)

def sample_and_save_binary_images(model, episode, random_latent_vectors, device, step_id, save_dir='binary_images', threshold=0.5):
    # Ensure the save directory exists
    # print(f"current working directory: {os.getcwd()}")
    episode = 'pair'
    sample_folder = f"{save_dir}/worlds_{episode}"
    os.makedirs(sample_folder, exist_ok=True)

    # os.makedirs(save_dir, exist_ok=True)

    # Sample random vectors from the standard normal distribution
    # random_latent_vectors = torch.randn(num_samples, latent_dim).to(device)
    # sample random verctor to be (0,1) range
    # random_latent_vectors = torch.rand(num_samples, latent_dim).to(device)
    # Generate images from the random latent vectors
    with torch.no_grad():
        generated_images = model.decoder(random_latent_vectors)

    # Reshape the output to image format
    generated_images = generated_images.view(-1, 1, 30, 30)
    occupancy_rate_list = []
    occupancy_rate = 0
    for i, image in enumerate(generated_images):
        # Apply threshold to convert image to binary
        
        binary_image = (image.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        good, _ =find_path_for_robot(binary_image, robot_size=5)


        # calculate occupancy rate in the image
        occupancy_rate = np.sum(binary_image) / binary_image.size
        occupancy_rate_list.append(occupancy_rate)
        # Save each binary image as .npy file
        
        if not good:
            # print(f"bad map: {episode}_{step_id}_{i}")     
            # random choose one from npy_files_list   
            binary_image = random.choice(npy_files_list)
        save_path = os.path.join(sample_folder, f"binary_image_{step_id}.npy")
        # print(f"Saving binary image to {save_path}")
        np.save(save_path, binary_image)
    # calculate statistics of occupancy rate
    # print(f"Occupancy rate mean: {np.mean(occupancy_rate_list)}")
    # print(f"Occupancy rate std: {np.std(occupancy_rate_list)}")
    # print(f"Occupancy rate min: {np.min(occupancy_rate_list)}")
    # print(f"Occupancy rate max: {np.max(occupancy_rate_list)}")
    return generated_images, occupancy_rate, good

# Usage example

if __name__ == "__main__":
    model_path = "vae_model_new.pth"
    num_samples = 100
    latent_dim = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    loaded_model = load_model(model_path, latent_dim, device)

    # Generate and visualize images
    # generate_images(loaded_model, num_samples, latent_dim, device)
    occupancy_rate_list = []
    for i in range(num_samples):
        random_latent_vectors = torch.rand(1, latent_dim).to(device)

        _, occupancy_rate = sample_and_save_binary_images(loaded_model, 'pair', random_latent_vectors, device, i, save_dir='binary_images', threshold=0.5)
        occupancy_rate_list.append(occupancy_rate)
    print(f"Occupancy rate mean: {np.mean(occupancy_rate_list)}")
    print(f"Occupancy rate std: {np.std(occupancy_rate_list)}")
    print(f"Occupancy rate min: {np.min(occupancy_rate_list)}")
    print(f"Occupancy rate max: {np.max(occupancy_rate_list)}")
    # sample_and_save_binary_images(loaded_model, num_samples, random_latent_vectors, device)
# model_path = "vae_model.pth"
# num_samples = 100
# latent_dim = 20
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model
# loaded_model = load_model(model_path, latent_dim, device)

# # Generate and visualize images
# # generate_images(loaded_model, num_samples, latent_dim, device)
# sample_and_save_binary_images(loaded_model, num_samples, latent_dim, device)

