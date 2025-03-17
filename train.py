# from Model import ALittleBitOfThisAndALittleBitOfThatNet
from model import encoder, decoder, merger, refiner
from loss import VoxelLoss
from Dataset import ShapeNet3DDataset
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
import torchvision.transforms as T

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



def initiate_training_environment(path: str):
    count = 0
    if os.path.isdir(path):
        count = len(os.listdir(path))
    else:
        os.mkdir(os.path.join(path))
    new_path = os.path.join(path, str(count))
    os.mkdir(new_path)

    return new_path


def train(configs):
    train_cfg = configs["train"]
    augmentation_cfg = configs["augmentation"]
    model_cfg = configs["model"]
    dataset_cfg = configs["dataset"]

    train_transformations = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.RandomApply([T.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.2),
        T.Lambda(lambda x: torch.clamp(x, 0.0, 255.0)),
        T.ColorJitter(brightness=augmentation_cfg["brightness"],
                      contrast=augmentation_cfg["contrast"],
                      saturation=augmentation_cfg["saturation"]),
        T.ToTensor()
    ])
    test_transformations = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])

    train_dataset = ShapeNet3DDataset(dataset_path=dataset_cfg["data_path"],
                                      json_file_path=dataset_cfg["json_mapper"],
                                      split='train',
                                      transforms=train_transformations)
    
    test_dataset = ShapeNet3DDataset(dataset_path=dataset_cfg["data_path"],
                                      json_file_path=dataset_cfg["json_mapper"],
                                      split='test',
                                      transforms=test_transformations)
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)


    Encoder = encoder.Encoder(pretrained=model_cfg["encoder"]["pretrained"]).to(configs["device"])
    Decoder = decoder.Decoder().to(configs["device"])
    Merger = merger.Merger(model_cfg["lrelu_factor"]).to(configs["device"])
    Refiner = refiner.Refiner(model_cfg["lrelu_factor"], model_cfg["use_bias"]).to(configs["device"])

    E_optim = torch.optim.Adam(Encoder.parameters(), lr=train_cfg["lr"])
    D_optim = torch.optim.Adam(Decoder.parameters(), lr=train_cfg["lr"])
    M_optim = torch.optim.Adam(Merger.parameters(), lr=train_cfg["lr"])
    R_optim = torch.optim.Adam(Refiner.parameters(), lr=train_cfg["lr"])

    for epoch in range(train_cfg["epochs"]):
        pass



# Main function to run training
def main():
    configs = None
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    train_path = initiate_training_environment(configs["train"]["output_dir"])
    configs["train_path"] = train_path
#     ##TODO: CALL THESE 2 NIGGAS AT THE END OF EACH EPOCH
#     train_dataloader.dataset.set_n_views_rendering(10) ## 12 is a random number from lower bound 2 upper bound in config file
#     train_dataloader.dataset.choose_images_indices_for_epoch()
#     # train_pix2vox(cfg, train_dataloader, train_dataloader)
    train(configs=configs)



if __name__ == "__main__":
    main()






