## Pix2VoxFormer Architecture
## [This trash didn't work, however i'm keeping it here. don't waste your time exploring this abs trash 🙏]
![Alt Text](images/Next%20SOTA%203D%20reconstruction%20model(4).svg)


improvements made to the original **Pix2Vox** architecture: 
- Using a version of **ConvNext** trained on **ImageNet22k** as  as an encoder to help with generalization and better reconstruction for unseen categories
- The encoder consists of both ConvNext and ViT such that: 
    - **ViT** captures the global context of the input image
    - **ConvNext** captures the local features of the input image
- the outputs of these are then concatenated, reshaped and upsampled using a decoder module 
- Introducing skip connections between the encoder and the decoder to perserve spatial features. So it looks more like a UNet now
    
- Training the model hasn't started yet, development is in progress 
- Expectations: 
    - Worst Case Scenario, the model will exhibit the features propagated from the encoder + the transformer, which will result in identical results to the original Pix2Vox model
    - Best Case Scenario: The model will benifit from the skip connections and the transformer's latent space and actually gain more accuracy
