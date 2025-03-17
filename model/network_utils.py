import torch.nn as nn
import torch.nn.init as init



def init_weights(l):
    if  (isinstance(l, nn.Conv2d) or isinstance(l, nn.Conv3d) or isinstance(l, nn.ConvTranspose3d)):
        init.kaiming_normal_(l.weight)
        if l.bias is not None:
            init.constant_(l.bias, 0)

    elif (isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm3d)):
        init.constant_(l.weight, 1)
        init.constant_(l.bias, 0)

    elif (isinstance(l, nn.Linear)):
        init.normal_(l.weight, 0, 0.01)
        init.constant_(l.bias, 0)