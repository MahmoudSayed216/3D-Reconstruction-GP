import torch.nn as nn

class VoxelLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(VoxelLoss, self).__init__()
        self.weight = weight
        self.loss_fn = nn.BCEWithLogitsLoss()  # Safe for AMP
        
    def forward(self, pred, target):
        return self.weight * self.loss_fn(pred, target)