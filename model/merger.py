import torch
import torch.nn as nn

from utils.debugger import DEBUG, DEBUGGER_SINGLETON
DEBUGGER_SINGLETON.active = True



class Merger(nn.Module):
    def __init__(self, lrelu_factor):
        super(Merger, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(9, 9, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(9, 9, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(9, 9, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )

        self.layer4 = nn.Sequential(
            nn.Conv3d(9, 9, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )

        self.layer5 = nn.Sequential(
            nn.Conv3d(36, 9, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )

        self.layer6 = nn.Sequential(
            nn.Conv3d(9, 1, 3, 1, 1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(lrelu_factor)
        )


    def forward(self, raw_features, coarse_volumes):
        n_view_rendering = raw_features.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_view_rendering):
            raw_feature = raw_features[i].squeeze(dim=1)

            vol1 = self.layer1(raw_features)
            vol2 = self.layer2(vol1)
            vol3 = self.layer3(vol2)
            vol4 = self.layer4(vol3)
            volumes_concatenated = torch.cat([vol1, vol2, vol3, vol4], dim=1)
            vol = self.layer5(volumes_concatenated)

            vol = self.layer6(vol)
            vol = vol.squeeze(dim=1)
            volume_weights.append(vol)

        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)
        DEBUG("volume size", volume_weights.shape)
        DEBUG("coarse size", coarse_volumes.shape)

        coarse_volumes = coarse_volumes*volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)

        return torch.clamp(coarse_volumes, min=0, max=1)
