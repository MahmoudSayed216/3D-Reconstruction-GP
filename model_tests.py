from model.encoder import Encoder
from Dataset import ShapeNet3DDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from Model import ALittleBitOfThisAndALittleBitOfThatNet
import torch

transformations = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
    T.RandomApply([T.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.2),
    T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
    T.ToTensor(),
    T.ColorJitter()
])




train_dataset = ShapeNet3DDataset(
    "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/",
    "/home/mahmoud-sayed/Desktop/Graduation Project/current/Pix2VoxFormer/dataset mapper/ShapeNet3DClone.json",
    split='train',
    transforms=transformations
    )
train_dataset.set_n_views_rendering(10)
train_dataset.choose_images_indices_for_epoch()
data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


model = ALittleBitOfThisAndALittleBitOfThatNet(device="cpu", lrelu_factor=0.2,pretrained=False)
print(sum(p.numel() for p in model.parameters()))

for idx, data in enumerate(data_loader):
    v, r, l = data
    model(v, r) 