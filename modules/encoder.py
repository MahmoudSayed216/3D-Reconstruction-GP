# Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTEncoder(nn.Module):
    def __init__(self):
        """
        Vision Transformer Encoder for AbdeinNet.
        
        Args:
            pretrained (bool): Whether to use pretrained weights
        """
        super(ViTEncoder, self).__init__()
        
        # Load pretrained ViT

        self.vit = vit_b_16(weights=None)

        # Remove the classification head
        self.vit.heads = nn.Identity()
        
        # Feature dimension from ViT
        self.features_dim = 768
        
        # Projection layer to match the decoder's expected input
        self.projection = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512)
            ##TODO: add batch norm
        )
        
    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            torch.Tensor: Encoded features
        """
        # Process through ViT
        batch_size = x.size(0)
        features = self.vit(x)  # [batch_size, features_dim]
        
        # Project features
        features = self.projection(features)  # [batch_size, 512]
        
        return features
