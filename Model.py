import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.merger import Merger


class DingDongNet(nn.Module):
    def __init__(self, device, lrelu_factor,  pretrained=True):
        super(DingDongNet, self).__init__()
        self.encoder = Encoder(device, pretrained)
        self.decoder = Decoder()
        self.merger = Merger(lrelu_factor)

    def forward(self, v_imgs, r_imgs):
        batch_size = v_imgs.shape[0]
        n_views = v_imgs.shape[1]

        # feading inputs to encoder
        lvl0_fms, lvl1_fms, lvl2_fms, lvl3_fms, latent_space = self.encoder(v_imgs, r_imgs)


        # reshaping feature maps and concatenating them
        latent_space = latent_space.view(batch_size, n_views, 128, 2,2,2)
        lvl3_fms = lvl3_fms.view(batch_size, n_views, 1568, 2, 2, 2)
        base_input = torch.cat([lvl3_fms, latent_space], dim=2)

        # feading features to decoder
        raw, gen = self.decoder(lvl0_fms, lvl1_fms, lvl2_fms, base_input)

        volume = self.merger(raw, gen)

