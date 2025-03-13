from model.encoder import Encoder
from Dataset import ShapeNet3DDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader



transformations = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
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



enc = Encoder(device="cpu", pretrained=False)
print(sum(p.numel() for p in enc.parameters()))
for idx, data in enumerate(data_loader):
    v, r, l = data
    enc(v, r) 