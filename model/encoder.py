import torch
import torch.nn as nn
from torchvision.models import resnet50, vit_b_16


class Encoder(nn.Module):

    def configure_vit(self, pretrained = True):
        vit = vit_b_16(pretrained=False)

        vit = vit.to(device= "cuda:0" if self.device == "cuda" else "cpu")
        vit.heads = nn.Identity()

        for param in vit.parameters():
            param.requires_grad = False

        self.features_dim = 768
        self.latent_space_size = 1024
        
        # Projection layer to match the decoder's expected input
        projection = nn.Sequential(
            nn.Linear(self.features_dim, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2),
            nn.Linear(768, self.latent_space_size),
            nn.BatchNorm1d(self.latent_space_size),
            nn.LeakyReLU(0.2),
        )
        return vit, projection
    

    def configure_resnet(self, pretrained = True):
        resnet = resnet50(pretrained=True)
        resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2
        ])
        for param in resnet.parameters():
            param.requires_grad = False
        resnet = resnet.to(device= "cuda:1" if self.device == "cuda" else "cpu")

        return resnet


    def __init__(self, device, pretrained=True):
        super(Encoder, self).__init__()
        self.device = device
        self.ViT, self.projection = self.configure_vit(pretrained = pretrained)
        self.ResNet = self.configure_resnet(pretrained = pretrained)
       ## 1x1 conv to lower the number of params in the up projection
       ## resnet feature maps are probably unneeded 
        self.layer0 = nn.Sequential(*[
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
        # print("original input size: ",img.shape)
        x = self.ResNet(img)
        # print("after resnet: ", x.shape)
        lvl0 = self.layer0(x)
        # print("after layer 0: ", x.shape)
        lvl1 = self.layer1(lvl0)
        # print("after layer 1: ", x.shape)
        lvl2 = self.layer2(lvl1)
        # print("after layer 2: ", x.shape)
        lvl3 = self.layer3(lvl2)
        # print("after layer 3: ", x.shape)
        return lvl0, lvl1, lvl2, lvl3



    def forward_ViT(self, img):
        x = self.ViT(img)
        # print("after vit: ", x.shape)
        x = self.projection(x)
        # print("after projection: ", x.shape)
        return x

    
    def forward(self, v_imgs, r_imgs):
        r_imgs = r_imgs.permute(1, 0, 2, 3, 4).contiguous()
        v_imgs = v_imgs.permute(1, 0, 2, 3, 4).contiguous()
        level_3_features = []
        level_2_features = []
        level_1_features = []
        level_0_features = []
        vit_outputs = []
        for image in r_imgs:
            lvl0, lvl1, lvl2, lvl3 = self.forward_cnn(image)
            level_0_features.append(lvl0)
            level_1_features.append(lvl1)
            level_2_features.append(lvl2)
            level_3_features.append(lvl3)
        for image in v_imgs:
            v = self.forward_ViT(image)
            vit_outputs.append(v)
        
        level_0_features = torch.stack(level_0_features).permute(1, 0, 2, 3, 4).contiguous()
        level_1_features = torch.stack(level_1_features).permute(1, 0, 2, 3, 4).contiguous()
        level_2_features = torch.stack(level_2_features).permute(1, 0, 2, 3, 4).contiguous()
        level_3_features = torch.stack(level_3_features).permute(1, 0, 2, 3, 4).contiguous()
        vit_outputs = torch.stack(vit_outputs).permute(1, 0, 2)

        print(level_0_features.shape)
        print(level_1_features.shape)
        print(level_2_features.shape)
        print(level_3_features.shape)
        print(vit_outputs.shape)
        print("____")

        return level_0_features, level_1_features, level_2_features, level_3_features, vit_outputs