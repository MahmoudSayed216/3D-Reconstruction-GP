
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
