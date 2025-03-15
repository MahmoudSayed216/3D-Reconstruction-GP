## TODO: a function that takes a 2D tensor of size H,W and reshapes it into X,Y,Z s.t. X = Y = Z
import torch
import torch.nn as nn
from utils.debugger import LOG, DEBUG, DEBUGGER_SINGLETON
DEBUGGER_SINGLETON.active =True


class ConvTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTConv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, padding=1, kernel_size=3),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        ])

        self.tconv = nn.Sequential(*[
            nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU()
        ])

    def forward(self, x):
        x = self.conv(x)
        x = self.tconv(x)

        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ## initially, the size is:      1696,2,2,2 -> latent space (+) bottleneck
        self.upsample1 = ConvTConv(in_channels=1696, out_channels=512)
        ## after this, the size is:     512,4,4,4  -> level 2 (+) upsample1 
        self.upsample2 = ConvTConv(in_channels=512, out_channels=128)
        ## after this, the size is:     128,8,8,8
        self.upsample3 = ConvTConv(in_channels=226, out_channels=16)
        ## after this, the size is:     32,16,16,16  -> level 1 (+) upsample3
        self.upsample4 = ConvTConv(in_channels=114, out_channels=8)
        ## after this, the size is:     8,32,32,32 -> level 0 (+) upsample4
        self.upsample5 = nn.Sequential(*[
            nn.ConvTranspose3d(in_channels=57, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ])
        

        



    def reshape_feature_maps(self,fms, input_side_length, output_size_length):
        batch_size = fms.shape[0]
        feature_maps_size = fms.shape[1]
        factor = input_side_length**2/output_size_length**3 ## 
        new_fm_size = int(feature_maps_size*factor)
        fms = fms.view(batch_size, new_fm_size, output_size_length, output_size_length, output_size_length)
        return fms

    def concatenate_feature_maps(self, fm1, fm2):
        return torch.cat([fm1, fm2], dim=1)


    # base input is level3 concatenated with latent_space
    def forward(self, level_0, level_1, level_2, base_input):
        base_input = base_input.permute(1, 0, 2, 3, 4, 5).contiguous()
        level_2 = level_2.permute(1, 0, 2, 3, 4).contiguous()
        level_1 = level_1.permute(1, 0, 2, 3, 4).contiguous()
        level_0 = level_0.permute(1, 0, 2, 3, 4).contiguous()
        raw_features = []
        gen_volumes = []
        for i, fm in enumerate(base_input):

            output = self.upsample1(fm)
            output = self.upsample2(output)

            lvl2_view_features = level_2[i]
            lvl2_view_features = self.reshape_feature_maps(lvl2_view_features, 14, 8)
            lvl2_view_features = self.concatenate_feature_maps(lvl2_view_features, output)
            output = self.upsample3(lvl2_view_features)


            lvl1_view_features = level_1[i]
            lvl1_view_features = self.reshape_feature_maps(lvl1_view_features, 28, 16)
            lvl1_view_features = self.concatenate_feature_maps(lvl1_view_features, output)
            output = self.upsample4(lvl1_view_features)
            raw_feature = output


            lvl0_view_features = level_0[i]
            lvl0_view_features = self.reshape_feature_maps(lvl0_view_features, 28, 32)
            lvl0_view_features = self.concatenate_feature_maps(lvl0_view_features, output)
            output = self.upsample5(lvl0_view_features)
            
            
            raw_feature = torch.cat([raw_feature, output], dim=1)
            gen_volumes.append(torch.squeeze(output, dim=1))
            raw_features.append(raw_feature)


        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()

        return raw_feature, gen_volumes