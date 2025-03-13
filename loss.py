import torch.nn as nn

class VoxelLoss(nn.Module):
    def __init__(self, weight=1.0):
        """
        Voxel loss function.
        
        Args:
            weight (float): Weight of the loss
        """
        super(VoxelLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        """
        Forward pass of the loss function.
        
        Args:
            pred (torch.Tensor): Predicted voxels of shape (batch_size, 1, D, H, W)
            target (torch.Tensor): Target voxels of shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        batch_size = pred.size(0)
        
        # Binary cross entropy loss
        bce_loss = nn.BCELoss()(pred, target)
        
        return self.weight * bce_loss
