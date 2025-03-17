import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torchvision.models import resnet50, vit_b_16



class Encoder(nn.Module):

    def configure_vit(self, configs):
        if configs["encoder"]["pretrained"]:
            vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            vit = vit_b_16(weights = None)
        # vit = vit.to(device= "cuda:0" if self.device == "cuda" else "cpu")
        vit.heads = nn.Identity()

        for param in vit.parameters():
            param.requires_grad = False

        self.features_dim = configs["encoder"]["feature_dim"]
        self.middle_dim = configs["encoder"]["middle_dim"]
        self.latent_space_size = configs["encoder"]["latent_dim"]
        
        # Projection layer to match the decoder's expected input
        projection = nn.Sequential(
            nn.Linear(self.features_dim, self.middle_dim),
            nn.BatchNorm1d(self.middle_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.middle_dim, self.latent_space_size),
            nn.BatchNorm1d(self.latent_space_size),
            nn.LeakyReLU(0.2),
        )
        return vit, projection
    

    def configure_resnet(self, configs):
        if configs["encoder"]["pretrained"] == True:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(weights=None)
        resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2
        ])
        for param in resnet.parameters():
            param.requires_grad = False
        # resnet = resnet.to(device= "cuda:1" if self.device == "cuda" else "cpu")
        return resnet


    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.configs = configs
        self.ViT, self.projection = self.configure_vit(configs=configs)
        self.ResNet = self.configure_resnet(configs=configs)
       ## 1x1 conv to lower the number of params in the up projection
       ## resnet feature maps are probably unneeded 
        self.layer0 = nn.Sequential(*[
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        ])

        self.layer1 = nn.Sequential(*[
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.layer2 = nn.Sequential(*[
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        ])
        self.layer3 = nn.Sequential(*[
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        ])



    def forward_cnn(self, img):
        x = self.ResNet(img)
        lvl0 = self.layer0(x)
        lvl1 = self.layer1(lvl0)
        lvl2 = self.layer2(lvl1)
        lvl3 = self.layer3(lvl2)
        return lvl0, lvl1, lvl2, lvl3



    def forward_ViT(self, img):
        x = self.ViT(img)
        x = self.projection(x)
        return x

    
    def forward(self, v_imgs, r_imgs):
        r_imgs = r_imgs.permute(1, 0, 2, 3, 4).contiguous()
        v_imgs = v_imgs.permute(1, 0, 2, 3, 4).contiguous()
        r_imgs = torch.split(r_imgs, 1, dim=0)
        v_imgs = torch.split(v_imgs, 1, dim=0)

        level_3_features = []
        level_2_features = []
        level_1_features = []
        level_0_features = []
        vit_outputs = []
        for image in r_imgs:
            image = image.squeeze(dim=0)
            lvl0, lvl1, lvl2, lvl3 = self.forward_cnn(image)
            level_0_features.append(lvl0)
            level_1_features.append(lvl1)
            level_2_features.append(lvl2)
            level_3_features.append(lvl3)
        for image in v_imgs:
            image = image.squeeze(dim=0)
            v = self.forward_ViT(image)
            vit_outputs.append(v)
        
        level_0_features = torch.stack(level_0_features).permute(1, 0, 2, 3, 4).contiguous()
        level_1_features = torch.stack(level_1_features).permute(1, 0, 2, 3, 4).contiguous()
        level_2_features = torch.stack(level_2_features).permute(1, 0, 2, 3, 4).contiguous()
        level_3_features = torch.stack(level_3_features).permute(1, 0, 2, 3, 4).contiguous()
        vit_outputs = torch.stack(vit_outputs).permute(1, 0, 2)


        return level_0_features, level_1_features, level_2_features, level_3_features, vit_outputs