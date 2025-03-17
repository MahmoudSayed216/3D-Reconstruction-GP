from Model import ALittleBitOfThisAndALittleBitOfThatNet
from loss import VoxelLoss
from Dataset import ShapeNet3DDataset
# Training Loop and Loss Functions
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import os
import time
import numpy as np
from tensorboardX import SummaryWriter

## TODO: build a weight loading method to make it easier to trian on a different account without Fing up the code

def train_pix2vox(cfg, train_dataloader, val_dataloader):

    # Create model
    model = ALittleBitOfThisAndALittleBitOfThatNet(
        ef_dim=cfg['model']['ef_dim'],
        z_dim=cfg['model']['z_dim'],
        use_refiner=cfg['model']['use_refiner'],
        # pretrained_encoder=True
    ).to(cfg['device'])
    
    # Set up loss functions
    voxel_loss = VoxelLoss(weight=1.0).to(cfg['device'])
    
    # TODO: SET UP THE OPTIMIZERS IN A BETTER WAY
    # Set up optimizer
    # Don't include encoder in the parameter list if it's frozen
    if model.encoder.vit.parameters().__next__().requires_grad:
        params = model.parameters()
    else:
        # Only optimize decoder, refiner, merger and encoder's projection layer
        params = list(model.decoder.parameters()) + \
                (list(model.refiner.parameters()) if model.use_refiner else []) + \
                list(model.merger.parameters()) + \
                list(model.encoder.projection.parameters())
    
    optimizer = optim.Adam(
        params=params,
        lr=cfg['train']['lr'],
        betas=(cfg['train']['beta1'], 0.999)
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg['train']['lr_milestones'],
        gamma=cfg['train']['gamma']
    )
    
    # Set up tensorboard writer
    output_dir = os.path.join(cfg['train']['output_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    # Train the model
    print(f"Starting training for {cfg['train']['epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg['train']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            imgs = batch[0].to(cfg['device'])  # [batch_size, n_views, C, H, W]
            voxels = batch[1].unsqueeze(1).to(cfg['device'])  # [batch_size, 1, D, H, W]
            
            # Forward pass
            outputs = model(imgs, voxels)
            
            # Compute losses
            raw_loss = voxel_loss(outputs['raw_voxels'].mean(dim=1), voxels)
            coarse_loss = voxel_loss(outputs['coarse_voxels'], voxels)
            merged_loss = voxel_loss(outputs['merged_voxels'], voxels)
            
            # Total loss
            loss = raw_loss + coarse_loss + merged_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            print(train_loss)
            
            # Print progress
            if (batch_idx + 1) % cfg['train']['print_every'] == 0:
                print(f"Epoch [{epoch+1}/{cfg['train']['epochs']}] "
                      f"Batch [{batch_idx+1}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        #TODO: MAKE AN INDEPENDENT VALIDATION FUNCTION def validate
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_dataloader:
                imgs = batch['imgs'].to(cfg['device'])
                voxels = batch['voxel'].unsqueeze(1).to(cfg['device'])
                
                outputs = model(imgs)
                
                # Only compute loss on the final output
                loss = voxel_loss(outputs['merged_voxels'], voxels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % cfg['train']['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(output_dir, f'last.pth'))
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{cfg['train']['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Time: {time.time() - start_time:.2f}s")
    
    print("Training completed!")
    writer.close()

def initiate_environment(path: str):
    count = 0
    if os.path.isdir(path):
        count = len(os.listdir(path))
    else:
        os.mkdir(os.path.join(path))
    new_path = os.path.join(path, str(count))
    os.mkdir(new_path)



# Main function to run training
def main():
    configs = None
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    initiate_environment(configs["train"]["output_dir"])
#     # Create dataset and dataloader
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
#     ])
    
#     train_dataset = ShapeNet3DDataset(cfg["dataset"]["data_path"], cfg["dataset"]["config_path"], 'train', transform)

#     # val_dataset = ShapeNetDataset(
#     #     dataset_path=cfg['dataset']['val_path'],
#     #     split='val',
#     #     categories=cfg['dataset']['categories'],
#     #     transforms=transform
#     # )
    
#     train_dataloader = DataLoader(
#         dataset=train_dataset,
#         batch_size=20,
#         shuffle=True,
#         # num_workers=cfg['train']['num_workers']
#     )

#     ##TODO: CALL THESE 2 NIGGAS AT THE END OF EACH EPOCH
#     train_dataloader.dataset.set_n_views_rendering(10) ## 12 is a random number from lower bound 2 upper bound in config file
#     train_dataloader.dataset.choose_images_indices_for_epoch()
#     # images, model= train_dataset[0]
#     start = time.time()
#     iter = train_dataloader._get_iterator()
#     i, m = iter._next_data()
#     end = time.time()
#     print(end-start)
#     print(i.shape)
#     print(m.shape)
#     # val_dataloader = DataLoader(
#     #     dataset=val_dataset,
#     #     batch_size=cfg['train']['batch_size'],
#     #     shuffle=False,
#     #     num_workers=cfg['train']['num_workers']
#     # )

#     # train_pix2vox(cfg, train_dataloader, train_dataloader)




if __name__ == "__main__":
    main()








# # train_transforms = utils.data_transforms.Compose([
# #     utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
# #     utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
# #     utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
# #     utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
# #     utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
# #     utils.data_transforms.RandomFlip(),
# #     utils.data_transforms.RandomPermuteRGB(),
# #     utils.data_transforms.ToTensor(),
# # ])