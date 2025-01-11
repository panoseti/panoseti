
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.gelu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_op1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.conv_op2 = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_op1(x)
        out2 = self.conv_op2(out1)
        return out2

class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv_op = DownConv(in_channels=in_channels, out_channels=out_channels)
        self.conv_op = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_conv_op(x)
        out = self.conv_op(out)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_inv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.gelu = nn.GELU()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_inv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv_op = UpConv(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.conv_block = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up_conv_op(x)
        out = self.conv_block(out)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.linear_1(x)
        out2 = self.gelu(out1)
        out3 = self.linear_2(out2)
        return out3


class Unflatten16x16(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_inv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_inv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x



class BetaVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=16, latent_dim=8):
        """
        Beta-VAE model for 16x16 grayscale images.
        
        Args:
            hidden_dim (int): Number of features in the intermediate layers.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),     # ouput shape: (hidden_dim, 16, 16)
            DownBlock(hidden_dim, hidden_dim),     # output shape: (hidden_dim, 8, 8)
            DownBlock(hidden_dim, hidden_dim * 2), # output shape: (hidden_dim * 2, 4, 4)
            DownBlock(hidden_dim * 2, hidden_dim * 2), # output shape: (hidden_dim * 4, 2, 2)
        )
        self.fc_mu = nn.Linear(hidden_dim * 2 * 2 * 2, latent_dim) # Notice that latent_dim controls the  information bottleneck
        self.fc_logvar = nn.Linear(hidden_dim * 2 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 2 * 2 * 2)
        self.decoder = nn.Sequential(
            UpBlock(hidden_dim * 2, hidden_dim * 2), # output shape: (hidden_dim * 2, 4, 4)
            UpBlock(hidden_dim * 2, hidden_dim), # output shape: (hidden_dim, 8, 8)
            UpBlock(hidden_dim, hidden_dim),     # output shape: (hidden_dim, 16, 16)
            ConvBlock(hidden_dim, hidden_dim),   # output shape: (hidden_dim, 16, 16)
            nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1), # output shape: (1, 16, 16)
            # nn.Tanh(),
            # nn.Sigmoid(),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, 2, 2)  # Reshape to image
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def beta_vae_loss_function(recon_x, x, mu, logvar, beta=1.0, sparsity_weight=0.0, verbose=False):
    """
    Compute the loss for Beta-VAE with optional sparsity regularization.
    
    Args:
        recon_x (torch.Tensor): Reconstructed images.
        x (torch.Tensor): Original images.
        mu (torch.Tensor): Latent mean.
        logvar (torch.Tensor): Latent log variance.
        beta (float): Weight for the KL divergence term.
        sparsity_weight (float): Weight for the L1 sparsity penalty on latent variables.
    
    Returns:
        torch.Tensor: Total loss (scalar).
    """
    # reconstruction loss via mse objective 
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # l1 sparsity regularization 
    sparsity_loss = sparsity_weight * torch.sum(torch.abs(mu))
    
    # Total loss
    if verbose:
      print("recon_loss={0:0.6}, kl_div={1:0.4}, kv_div_loss_term: {2:0.4}, sparsity_loss={3:0.4}, sum(|mu|)={4:0.4}".format(recon_loss,  kl_div, beta * kl_div, sparsity_loss, torch.sum(torch.abs(mu))))
    total_loss = recon_loss + beta * kl_div + sparsity_loss
    return total_loss






class Unflatten32x32(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_inv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            stride=7,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_inv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

# Define the VAE model
class VaeModel32x32(nn.Module):
    def __init__(self, in_channels: int, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # Encoder
        self.conv_block_input = ConvBlock(in_channels, hidden_dim)
        self.down_block_32_to_16 = DownBlock(hidden_dim, hidden_dim)
        self.down_block_16_to_8 = DownBlock(hidden_dim, 2*hidden_dim)
        self.flatten = Flatten()
        self.fc_mu = FCBlock(2 * hidden_dim, latent_dim)
        self.fc_logvar = FCBlock(2 * hidden_dim, latent_dim)

        # Decoder
        self.fc_dec = FCBlock(latent_dim, 2 * hidden_dim * 8**2)
        self.unflatten = Unflatten32x32(2*hidden_dim)
        self.up_block_8_to_16 = UpBlock(2*hidden_dim, hidden_dim)
        self.up_block_16_to_32 = UpBlock(hidden_dim, hidden_dim)
        self.conv_block_output = ConvBlock(hidden_dim, hidden_dim)
        self.conv_op_311 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
    def encode(self, x):
        """Compresses x into a latent space representation."""
        # Forward through U-Net
        # print(x.shape)
        conv_block_1 = self.conv_block_input(x)
        down_1 = self.down_block_32_to_16(conv_block_1)
        down_2 = self.down_block_16_to_8(down_1)
        flatten = self.flatten(down_2)
        flatten = flatten.view(flatten.size(0), -1)
        mu = self.fc_mu(flatten)
        logvar = self.fc_logvar(flatten)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Separate the random sample from the learnable parameters."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def decode(self, z):
        x = self.fc_dec(z)
        # Decode 
        x = x.view(x.size(0), 2*self.hidden_dim, 8, 8)
        unflatten = self.unflatten(x)
        up_1 = self.up_block_8_to_16(unflatten)
        up_2 = self.up_block_16_to_32(up_1)
        conv_block_2 = self.conv_block_output(up_2)
        recon_x = self.conv_op_311(conv_block_2)
        print('x recon shape:', recon_x.shape)
        return recon_x
    
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




