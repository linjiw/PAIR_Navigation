import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Initialize a new wandb run
wandb.init(project="vae_project")

# Import the VAE and dataset classes (ensure these are accessible)
from model import VAE  # Replace 'vae_model' with the actual module name
from dataset import NumpyDataset  # Replace 'dataset' with the actual module name

# Define paths and parameters
train_data_path = "./dataset/train"  # Replace with your training data path
test_data_path = "./dataset/test"    # Replace with your testing data path
num_epochs = 1000
batch_size = 64
learning_rate = 1e-3
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_interval = 10
early_stopping_patience = 20  # Define patience for early stopping
early_stopping_threshold = 0.01  # Define threshold for considering test loss increase as overfitting

# Define the VAE model
model = VAE(latent_dim=latent_dim).to(device)

# Define the optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Load datasets
train_dataset = NumpyDataset(train_data_path)
test_dataset = NumpyDataset(test_data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 30*30), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
# Early stopping variables
best_test_loss = float('inf')
epochs_without_improvement = 0
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")

    train_loss /= len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {train_loss:.4f}')
    scheduler.step(train_loss)  # Step the scheduler based on the average training loss
    wandb.log({"epoch": epoch, "train_loss": train_loss})

# Evaluation loop
# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()

#             if epoch == num_epochs:
#                 n = min(data.size(0), 8)
#                 # Reshape data to 4D tensor for image grid
#                 data_reshaped = data.view(-1, 1, 30, 30)[:n]
#                 recon_reshaped = recon_batch.view(-1, 1, 30, 30)[:n]
#                 comparison = torch.cat([data_reshaped, recon_reshaped], dim=0)
#                 save_image(comparison.cpu(), f"reconstruction_{epoch}.png", nrow=n)
#                 wandb.log({"comparison": [wandb.Image(comparison.cpu(), caption=f"Reconstruction Epoch {epoch}")]})

#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#     wandb.log({"epoch": epoch, "test_loss": test_loss})
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    wandb.log({"epoch": epoch, "test_loss": test_loss})

    global best_test_loss, epochs_without_improvement

    # Check if test loss has increased beyond threshold
    if test_loss < best_test_loss - early_stopping_threshold:
        best_test_loss = test_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch} as test loss increased for {early_stopping_patience} consecutive epochs.')
            return True  # Return True to stop training

    return False  # Return False to continue training

# Run the training and testing
# Run the training and testing with early stopping
for epoch in range(1, num_epochs + 1):
    train(epoch)
    if test(epoch):
        break

# Define the file path where you want to save the model
model_save_path = "vae_model.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)
