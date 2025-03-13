import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinerNetwork(nn.Module):
    def __init__(self, ef_dim=32):
        """
        Refiner Network for AbdeinNet.
        
        Args:
            ef_dim (int): Number of filters in the first layer
        """
        super(RefinerNetwork, self).__init__()
        
        # 3D Convolutions
        self.conv1 = nn.Conv3d(1, ef_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ef_dim)
        
        self.conv2 = nn.Conv3d(ef_dim, ef_dim * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ef_dim * 2)
        
        self.conv3 = nn.Conv3d(ef_dim * 2, ef_dim * 4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(ef_dim * 4)
        
        self.conv4 = nn.Conv3d(ef_dim * 4, ef_dim * 8, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(ef_dim * 8)
        
        # 3D Transposed Convolutions
        self.deconv1 = nn.ConvTranspose3d(ef_dim * 8, ef_dim * 4, kernel_size=3, padding=1, bias=False)
        self.debn1 = nn.BatchNorm3d(ef_dim * 4)
        
        self.deconv2 = nn.ConvTranspose3d(ef_dim * 4, ef_dim * 2, kernel_size=3, padding=1, bias=False)
        self.debn2 = nn.BatchNorm3d(ef_dim * 2)
        
        self.deconv3 = nn.ConvTranspose3d(ef_dim * 2, ef_dim, kernel_size=3, padding=1, bias=False)
        self.debn3 = nn.BatchNorm3d(ef_dim)
        
        self.deconv4 = nn.ConvTranspose3d(ef_dim, 1, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the refiner.
        
        Args:
            x (torch.Tensor): Coarse voxel reconstruction of shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Refined voxel reconstruction
        """
        batch_size = x.size(0)
        
        # Encoder
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)  # [batch_size, ef_dim, 32, 32, 32]
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)  # [batch_size, ef_dim*2, 32, 32, 32]
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)  # [batch_size, ef_dim*4, 32, 32, 32]
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)  # [batch_size, ef_dim*8, 32, 32, 32]
        
        # Decoder
        x = F.leaky_relu(self.debn1(self.deconv1(x)), negative_slope=0.2)  # [batch_size, ef_dim*4, 32, 32, 32]
        x = F.leaky_relu(self.debn2(self.deconv2(x)), negative_slope=0.2)  # [batch_size, ef_dim*2, 32, 32, 32]
        x = F.leaky_relu(self.debn3(self.deconv3(x)), negative_slope=0.2)  # [batch_size, ef_dim, 32, 32, 32]
        x = torch.sigmoid(self.deconv4(x))  # [batch_size, 1, 32, 32, 32]
        
        return x
