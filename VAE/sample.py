

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from model import VAE  # Replace 'vae_model' with the actual module name
from dataset import NumpyDataset  # Replace 'dataset' with the actual module name
import torch
import os
import numpy as np

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

def sample_and_save_binary_images(model, num_samples, latent_dim, device, save_dir='binary_images', threshold=0.5):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Sample random vectors from the standard normal distribution
    # random_latent_vectors = torch.randn(num_samples, latent_dim).to(device)
    # sample random verctor to be (0,1) range
    random_latent_vectors = torch.rand(num_samples, latent_dim).to(device)
    # Generate images from the random latent vectors
    with torch.no_grad():
        generated_images = model.decoder(random_latent_vectors)

    # Reshape the output to image format
    generated_images = generated_images.view(-1, 1, 30, 30)
    occupancy_rate_list = []
    for i, image in enumerate(generated_images):
        # Apply threshold to convert image to binary
        binary_image = (image.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        # calculate occupancy rate in the image
        occupancy_rate = np.sum(binary_image) / binary_image.size
        occupancy_rate_list.append(occupancy_rate)
        # Save each binary image as .npy file
        save_path = os.path.join(save_dir, f"binary_image_{i}.npy")
        np.save(save_path, binary_image)
    # calculate statistics of occupancy rate
    print(f"Occupancy rate min: {np.min(occupancy_rate_list)}")
    print(f"Occupancy rate max: {np.max(occupancy_rate_list)}")
    print(f"Occupancy rate mean: {np.mean(occupancy_rate_list)}")
    print(f"Occupancy rate std: {np.std(occupancy_rate_list)}")

    return generated_images

# Usage example
model_path = "vae_model.pth"
num_samples = 100
latent_dim = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
loaded_model = load_model(model_path, latent_dim, device)

# Generate and visualize images
# generate_images(loaded_model, num_samples, latent_dim, device)
sample_and_save_binary_images(loaded_model, num_samples, latent_dim, device)

