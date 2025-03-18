# import torch

# raw_features = torch.rand((4, 3, 2, 1,1,1))
# print(raw_features.shape)
# raw_features = torch.split(raw_features, 1, dim=1)
# print(raw_features)
# print(len(raw_features))
# print(len(raw_features[0]))
# print(len(raw_features[0][0]))
# print(len(raw_features[0][0][0]))
# print(len(raw_features[0][0][0][0]))
# print(len(raw_features[0][0][0][0][0]))
# print(len(raw_features[0][0][0][0][0][0]))


# c = "D"

# print(c.lower())

# import torch
# o = torch.optim.Adam([])

# print(type(o))

# from Model import ALittleBitOfThisAndALittleBitOfThatNet
# from fvcore.nn import FlopCountAnalysis
# import torch
# model = ALittleBitOfThisAndALittleBitOfThatNet(device="cpu", lrelu_factor=0.2,pretrained=False)
# v_input_tensor = torch.rand((2, 10, 3, 224, 224))
# r_input_tensor = torch.rand((2, 10, 3, 224, 224))
# model.train()
# t_flops = FlopCountAnalysis(model, (v_input_tensor, r_input_tensor))
# model.eval()
# e_flops = FlopCountAnalysis(model, (v_input_tensor, r_input_tensor))

# print(f"Total training FLOPs: {t_flops.total()/(2*1e9):.2f} GFLOPs")  # Convert to GFLOPs (billion FLOPs)
# print(f"Total testing FLOPs: {e_flops.total()/(2*1e9):.2f} GFLOPs")  # Convert to GFLOPs (billion FLOPs)


# import torchinfo
# import torch
# from Model import ALittleBitOfThisAndALittleBitOfThatNet



# model = ALittleBitOfThisAndALittleBitOfThatNet("cpu", 0.2, False)

# torchinfo.summary(model, input_data=(torch.randn(1, 5, 3, 224, 224), torch.randn(1, 5, 3, 224, 224)))


# Total params: 171,046,351
# Trainable params: 83,802,767
# Non-trainable params: 87,243,584
# Total mult-adds (G): 39.62
# ==============================================================================================================
# Input size (MB): 6.02
# Forward/backward pass size (MB): 1555.69
# Params size (MB): 570.18
# Estimated Total Size (MB): 2131.89
#######################################################################################
# Total params: 171,046,351
# Trainable params: 83,802,767
# Non-trainable params: 87,243,584
# Total mult-adds (G): 77.24
# ==============================================================================================================
# Input size (MB): 12.04
# Forward/backward pass size (MB): 3083.48
# Params size (MB): 570.18
# Estimated Total Size (MB): 3665.71


# from tensorboard import program

# log_dir = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Pix2VoxFormer/outputs/2025-03-17_06-01-50/events.out.tfevents.1742184110.Aspire-A514"  # Adjust to your log directory
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', log_dir])
# tb.launch()


# import mlflow

# mlflow.start_run()

# mlflow.log_param("learning_rate", 0.001)
# mlflow.log_metric("loss", 0.45)
# mlflow.end_run()
# cpu = True
# if cpu:
#     from torch.cpu.amp import autocast, GradScaler
# else:
#     from torch.cuda.amp import autocast , GradScaler

# print(autocast)
# print(GradScaler)


# from utils.utils import visualize_from_file

# visualize_from_file("/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/ShapeNetVox32/ShapeNetVox32/02828884/1a55e418c61e9dab6f4a86fe50d4c8f0//model.binvox")



# # import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# def gaussian_random(low=1, high=12):
#     mu=6.5
#     sigma=3.5
#     while True:
#         x = np.random.normal(mu, sigma)  # Generate a Gaussian sample
#         if low <= x <= high:  # Accept only if within range
#             return int(round(x))  # Convert to integer

# j = list(range(12))

# def sum_lists(l1, l2):
#     l = [0]*12
#     for i in range(12):
#         l[i] = (l1[i]+l2[i])/2
#     return l
# accumulative_list = [0]*12

# for k in range(1000):
#     freq = [0]*12
#     for i in range(350):
#         r = gaussian_random(1, 12)-1
#         freq[r]+=1
#     accumulative_list = sum_lists(accumulative_list, freq)

# plt.bar(j, height=accumulative_list)
# plt.show()
# print(freq)


import torch
import matplotlib.pyplot as plt
from timm import create_model
from torchvision import transforms
from PIL import Image


# Load a pretrained Swin model with feature extraction
swin = create_model("swin_base_patch4_window7_224", pretrained=True, features_only=True)
swin.eval()

# Load and preprocess an image
image = Image.open("/home/mahmoud-sayed/Desktop/Cat_August_2010-4.jpg").convert("RGB")  # Replace with an actual image
transform = transforms.Compose([
    transforms.Resize(256),  # Resize so the shorter side is 256
    transforms.CenterCrop(224),  # Crop to (224, 224) centered
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
])

image = transform(image).unsqueeze(0)
# image = ToTensor()(image).unsqueeze(0)  # Add batch dimension

# Get feature maps
with torch.no_grad():
    feature_maps = swin(image)

def visualize_feature_map(feature_map, num_channels=100):
    """
    Plots a few feature maps (channels) from a given Swin Transformer feature map.
    """
    feature_map = feature_map.squeeze(0)  # Remove batch dim (C, H, W)
    
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
    for i in range(num_channels):
        axes[i].imshow(feature_map[i].cpu().numpy(), cmap="viridis")  # Pick channel i
        axes[i].axis("off")
    plt.show()

# Visualize feature maps from each stage
for i, fmap in enumerate(feature_maps):
    print(f"Visualizing Stage {i+1} with shape {fmap.shape}")
    visualize_feature_map(fmap, num_channels=6)  # Show 6 feature maps
