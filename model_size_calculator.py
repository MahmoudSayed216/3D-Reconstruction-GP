# this method is a very accurate but not an exact calculation
from model import decoder, encoder, merger, refiner
import yaml

configs = None
with open("/kaggle/working/3D-Reconstruction-GP/config.yaml", "r") as f:
    configs = yaml.safe_load(f)

model_cfg = configs["model"]
Encoder = encoder.Encoder(configs=model_cfg).to(configs["device"])
Decoder = decoder.Decoder().to(configs["device"])
Merger = merger.Merger(model_cfg["lrelu_factor"]).to(configs["device"])
Refiner = refiner.Refiner(model_cfg["lrelu_factor"], model_cfg["use_bias"]).to(configs["device"])

sum_params = lambda M: sum(p.numel() for p in M.parameters()) 

parameters = sum([sum_params(M) for M in [Encoder, Decoder, Merger, Refiner]])


size_in_mb = parameters>>18 


print("Model size: ", size_in_mb)




# ## INFO TO BE ADDED TO THE PRESENTATION
# import torchinfo
# import torch
# from Model import ALittleBitOfThisAndALittleBitOfThatNet



# model = ALittleBitOfThisAndALittleBitOfThatNet("cpu", 0.2, False)

# torchinfo.summary(model, input_data=(torch.randn(1, 5, 3, 224, 224), torch.randn(1, 5, 3, 224, 224)))

