
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


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x


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

class Unflatten16x16(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_inv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_inv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


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

# Define the VAE model
class VaeModel16x16(nn.Module):
    def __init__(self, in_channels: int, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # Encoder
        self.conv_block_input = ConvBlock(in_channels, hidden_dim)
        self.down_block_16_to_8 = DownBlock(hidden_dim, 2*hidden_dim)
        self.flatten = Flatten()
        self.fc_mu = FCBlock(2 * hidden_dim, latent_dim)
        self.fc_logvar = FCBlock(2 * hidden_dim, latent_dim)

        # Decoder
        self.fc_dec = FCBlock(latent_dim, 2 * hidden_dim * 8**2)
        self.unflatten = Unflatten16x16(2*hidden_dim)
        self.up_block_8_to_16 = UpBlock(2*hidden_dim, hidden_dim)
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
        down_1 = self.down_block_16_to_8(conv_block_1)
        flatten = self.flatten(down_1)
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
        # print('up_1 shape', up_1.shape)
        conv_block_1 = self.conv_block_output(up_1)
        # print('conv_block_1 shape', conv_block_1.shape)
        recon_x = self.conv_op_311(conv_block_1)
        # print('x recon shape:', recon_x.shape)
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






