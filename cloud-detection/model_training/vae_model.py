
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE model
class VaeModel(nn.Module):
    def __init__(self, in_channels: int, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # Log variance of latent space
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # (C, 32, 32)
            # nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from N(0,1)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), 128, 4, 4)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# Loss function: Reconstruction loss + KL divergence
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div






