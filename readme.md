## Pix2VoxFormer Architecture

![Alt Text](images/Next%20SOTA%203D%20reconstruction%20model(4).svg)


improvements made to the original **Pix2Vox** architecture: 
- Using a version of **ConvNext** trained on **ImageNet22k** as  as an encoder to help with generalization and better reconstruction for unseen categories
- The encoder consists of both ConvNext and ViT such that: 
    - ViT captures the global context of the input image
    - ConvNext captures the local features of the input image
- the outputs of these are then concatenated, reshaped and upsampled using a decoder module 
- Introducing skip connections between the encoder and the decoder to perserve spatial features. So it looks more like a UNet now
    
- Training the model hasn't started yet, development is in progress