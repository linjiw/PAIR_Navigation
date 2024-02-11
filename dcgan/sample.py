import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np
import os
import argparse
from dcgan import Generator  # Assuming Generator class is defined in models.py

# Argument parser for command line arguments

def sample( opt):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Load the trained generator
    generator = Generator(opt.encoder_dim, opt.latent_dim, opt.img_size, opt.channels)
    generator.load_state_dict(torch.load(f"./saved_models/generator_epoch_{opt.model_epoch}.pt"))
    if cuda:
        generator.cuda()

    # Set the generator to evaluation mode
    generator.eval()

    # Function to generate and save images
    # def generate_images(n_samples, latent_dim, img_shape):
    #     z = Tensor(np.random.normal(0, 1, (n_samples, latent_dim)))
    #     gen_imgs = generator(z)
    #     os.makedirs("sampled_images", exist_ok=True)
    #     save_image(gen_imgs.data, f"sampled_images/sample_{opt.model_epoch}.png", nrow=int(np.sqrt(n_samples)), normalize=True)
    def generate_images(n_samples, latent_dim, img_shape):
        z = Tensor(np.random.normal(0, 1, (n_samples, latent_dim)))
        gen_imgs = generator(z)
        os.makedirs("sampled_images", exist_ok=True)

        for i, img in enumerate(gen_imgs.data):
            save_image(img, f"sampled_images/sample_{opt.model_epoch}_{i}.png", normalize=True)

    # Generate and save images
    generate_images(opt.n_samples, opt.latent_dim, (opt.channels, opt.img_size, opt.img_size))


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--model_epoch", type=int, default=1920, help="epoch of saved model to load")
    parser.add_argument("--encoder_dim", type=int, default=50, help="size of each image dimension")

    opt = parser.parse_args()
    # model_pth = './saved_models/generator_epoch_0.pt'
    sample(opt)