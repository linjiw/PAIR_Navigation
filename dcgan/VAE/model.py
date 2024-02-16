import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(30*30, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        # self.drop_out1 = nn.Dropout(0.3)
        self.fc22 = nn.Linear(400, latent_dim)
        # self.drop_out2 = nn.Dropout(0.3)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        # h1 = self.drop_out1(h1)
        return self.fc21(h1), self.fc22(h1)  # returns mu and logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        # self.drop_out3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(400, 30*30)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        # h3 = self.drop_out3(h3)
        return torch.sigmoid(self.fc4(h3))

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        # send z through a sigmoid to ensure values are between 0 and 1
        return torch.sigmoid(z)
        # return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 30*30))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


