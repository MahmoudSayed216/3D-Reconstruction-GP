import torch.nn as nn
import torch

from modules.encoder import ViTEncoder
from modules.decoder import Decoder
from modules.refiner import RefinerNetwork
from modules.merger import Merger


class AbdeinNet(nn.Module):
    ##TODO: DEBUG THIS WHOLE NETWOKR
    def __init__(self, ef_dim=32, z_dim=512, use_refiner=True):
        """
        AbdeinNet model with ViT encoder.
        
        Args:
            ef_dim (int): Number of filters in the first layer
            z_dim (int): Dimension of the encoded features
            use_refiner (bool): Whether to use the refiner network
            pretrained_encoder (bool): Whether to use pretrained weights for the encoder
        """
        super(AbdeinNet, self).__init__()
        
        self.encoder = ViTEncoder()
        self.decoder = Decoder(ef_dim=ef_dim, z_dim=z_dim)
        self.merger = Merger(ef_dim=ef_dim)
        self.refiner = RefinerNetwork(ef_dim=ef_dim) if use_refiner else None
        
        self.use_refiner = use_refiner
        
        # Freeze encoder weights if using pretrained
        for param in self.encoder.vit.parameters():
            param.requires_grad = False
            
            # Note: Only the projection layer in the encoder is trainable
            for param in self.encoder.projection.parameters():
                param.requires_grad = True
        
    def forward(self, imgs, gt_voxel=None):
        """
        Forward pass of the AbdeinNet model.
        
        Args:
            imgs (torch.Tensor): Input images of shape (batch_size, n_views, C, H, W)
            gt_voxel (torch.Tensor, optional): Ground truth voxel grid for guidance
            
        Returns:
            dict: Dictionary containing all outputs (raw, merged, refined voxels)
        """
        batch_size = imgs.size(0)
        n_views = imgs.size(1)
        
        # Process each view
        raw_features = []
        raw_voxels = []
        
        for v in range(n_views):
            view_features = self.encoder(imgs[:, v])  # [batch_size, z_dim]
            view_voxels = self.decoder(view_features)  # [batch_size, 1, 32, 32, 32]
            
            raw_features.append(view_features)
            raw_voxels.append(view_voxels)
        
        # Stack all raw predictions
        raw_voxels = torch.stack(raw_voxels, dim=1)  # [batch_size, n_views, 1, 32, 32, 32]
        #TODO: DEBUG HERE
        
        # Merge the raw voxels from multiple views
        merged_voxels = self.merger(raw_voxels, gt_voxel)  # [batch_size, 1, 32, 32, 32]
        
        # Apply refinement to the merged voxels if refiner is used
        if self.use_refiner:
            refined_voxels = self.refiner(merged_voxels)  # [batch_size, 1, 32, 32, 32]
        else:
            refined_voxels = merged_voxels
        
        return {
            'raw_voxels': raw_voxels,  # [batch_size, n_views, 1, 32, 32, 32]
            'merged_voxels': merged_voxels,  # [batch_size, 1, 32, 32, 32]
            'refined_voxels': refined_voxels,  # [batch_size, 1, 32, 32, 32]
        }