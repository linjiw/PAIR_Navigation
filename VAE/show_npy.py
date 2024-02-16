import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_and_save_binary_images(folder_path, save_path='grid_image.png', grid_size=(5, 5)):
    # List all .npy files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_list.sort()  # Optional, sorts the files by name

    # Create a figure for the grid
    plt.figure(figsize=(10, 10))
    grid = gridspec.GridSpec(grid_size[0], grid_size[1])
    grid.update(wspace=0.1, hspace=0.1)

    for i, file in enumerate(file_list):
        if i >= grid_size[0] * grid_size[1]:
            break  # Stop if the grid is full

        # Load .npy file and plot it
        img = np.load(os.path.join(folder_path, file))
        ax = plt.subplot(grid[i])
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.axis('off')

    # Save the grid image
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # Optionally display the grid
    plt.imshow(plt.imread(save_path))
    plt.axis('off')
    plt.show()

# Usage example
folder_path = 'binary_images'  # Path to the folder containing .npy files
show_and_save_binary_images(folder_path)
