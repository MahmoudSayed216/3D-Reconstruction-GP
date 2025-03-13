import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, ef_dim=32, z_dim=512):
        """
        Decoder for AbdeinNet.
        
        Args:
            ef_dim (int): Number of filters in the first layer
            z_dim (int): Dimension of the encoded features
        """
        super(Decoder, self).__init__()
        
        # Feature vector projection
        self.fc = nn.Linear(z_dim, ef_dim * 8 * 2 * 2 * 2)
        
        # 3D Transposed Convolutions
        self.deconv1 = nn.ConvTranspose3d(ef_dim * 8, ef_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ef_dim * 4)
        
        self.deconv2 = nn.ConvTranspose3d(ef_dim * 4, ef_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ef_dim * 2)
        
        self.deconv3 = nn.ConvTranspose3d(ef_dim * 2, ef_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(ef_dim)
        
        self.deconv4 = nn.ConvTranspose3d(ef_dim, 1, kernel_size=4, stride=2, padding=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (batch_size, z_dim)
            
        Returns:
            torch.Tensor: Reconstructed 3D voxel grid
        """
        batch_size = x.size(0)
        
        x = self.fc(x)
        x = x.view(batch_size, -1, 2, 2, 2)  # [batch_size, ef_dim*8, 2, 2, 2]
        
        x = F.leaky_relu(self.bn1(self.deconv1(x)), negative_slope=0.2)  # [batch_size, ef_dim*4, 4, 4, 4]
        x = F.leaky_relu(self.bn2(self.deconv2(x)), negative_slope=0.2)  # [batch_size, ef_dim*2, 8, 8, 8]
        x = F.leaky_relu(self.bn3(self.deconv3(x)), negative_slope=0.2)  # [batch_size, ef_dim, 16, 16, 16]
        x = torch.sigmoid(self.deconv4(x))  # [batch_size, 1, 32, 32, 32]
        
        return x

