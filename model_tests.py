from model.encoder import Encoder
from Dataset import ShapeNet3DDataset
import torchvision.transforms as T

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

data_loader = 

enc = Encoder()