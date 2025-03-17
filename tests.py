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


import torchinfo
import torch
from Model import ALittleBitOfThisAndALittleBitOfThatNet



model = ALittleBitOfThisAndALittleBitOfThatNet("cpu", 0.2, False)

torchinfo.summary(model, input_data=(torch.randn(1, 5, 3, 224, 224), torch.randn(1, 5, 3, 224, 224)))


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