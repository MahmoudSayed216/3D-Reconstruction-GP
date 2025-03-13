import torch
import torch.nn as nn
from torchvision.models import resnet50, vit_l_16, swa


class Encoder(nn.Module):

    def configure_vit(self, pretrained = True):
        vit = vit_l_16(pretrained=False)

        vit = vit.to(device= "cuda:0" if self.device == "cuda" else "cpu")
        self.vit.heads = nn.Identity()
        
        self.features_dim = 768
        
        # Projection layer to match the decoder's expected input
        self.projection = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512)
            ##TODO: add batch norm
        )
        
        return vit
    

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
        self.ViT = self.configure_vit(pretrained = pretrained)
        self.ResNet = self.configure_resnet(pretrained = pretrained)
       
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
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
        print(img.shape)
        x = self.ResNet(img)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        return x



    def forward_ViT(self, img):
        pass

    
    def forward(self, v_imgs, r_imgs):
        cnn_outputs = []
        vit_outputs = []
        for image in r_imgs:
            r = self.forward_cnn(r_imgs)
            cnn_outputs.append(r)
        for image in v_imgs:
            v = self.forward_ViT(v_imgs)
            vit_outputs.append(v)


        return 