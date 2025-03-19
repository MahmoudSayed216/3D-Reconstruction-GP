# from Model import ALittleBitOfThisAndALittleBitOfThatNet
from model import encoder, decoder, merger, refiner
from metrics.loss import VoxelLoss
from metrics.IoU import compute_iou
from utils import network_utils
from writer import Writer
from Dataset import ShapeNet3DDataset
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os
import time
import torchvision.transforms as T
from utils.debugger import DEBUG, LOG, DEBUGGER_SINGLETON
import numpy as np



if torch.cuda.is_available():
    LOG("CUDA DETECTED")
else:
    LOG("CPU DETECTED")



## TODO: log losses and IoUs at 50
## TODO: build a weight loading method to make it easier to trian on a different account without Fing up the code

def gaussian_random(low=1, high=12):
    mu = 6.5
    sigma = 3.5
    while True:
        x = np.random.normal(mu, sigma)  # Generate a Gaussian sample
        if low <= x <= high:  # Accept only if within range
            return int(round(x))  # Convert to integer



def update_dataset_configs(loader, multi_view):
    random_value = 1 # gets updated only if multi_view
    if multi_view:
        random_value = gaussian_random(1, 12)
    loader.dataset.set_n_views_rendering(random_value)
        
    loader.dataset.choose_images_indices_for_epoch()
    return random_value

def merge_feature_maps(BATCH_SIZE, n_views, lvl3, latent_space):
    latent_space = latent_space.view(BATCH_SIZE, n_views, 128, 2,2,2)
    lvl3 = lvl3.view(BATCH_SIZE, n_views, 1568, 2, 2, 2)
    base_input = torch.cat([lvl3, latent_space], dim=2)
    return base_input
    




def compute_validation_metrics(Encoder, Decoder, Merger, Refiner, loader, n_views, THRESHOLDS, USE_MERGER):
    Encoder.eval()
    Decoder.eval()
    Merger.eval()
    Refiner.eval()

    TEST_LOSS_ACCUMULATOR = 0
    BATCH_SIZE = loader.batch_size
    IOU_40 = []
    IOU_50 = []
    IOU_75 = []
    loss_fn = VoxelLoss(weight=10)
    with torch.no_grad():
        with torch.amp.autocast("cuda"):

            for batch_idx, batch in enumerate(loader):
                r_img, v_img, gt_vol = batch
                v_img = v_img.to("cuda")
                r_img = r_img.to("cuda")
                gt_vol = gt_vol.to("cuda")
                #ENCODER
                lvl0, lvl1, lvl2, lvl3, latent_space = Encoder(v_img, r_img)
                
                #DECODER
                base_input = merge_feature_maps(BATCH_SIZE, n_views, lvl3, latent_space)
                raw, gen_vol = Decoder(lvl0, lvl1, lvl2, base_input)

                #MERGER
                if USE_MERGER:
                    gen_vol = Merger(raw, gen_vol)
                else:
                    gen_vol = gen_vol.squeeze(dim = 1)
                #REFINER
                # if USE_REFINER:
                gen_vol = Refiner(gen_vol)

                gen_vol = gen_vol.squeeze(dim=1)

                loss = loss_fn(gen_vol, gt_vol)
                TEST_LOSS_ACCUMULATOR+=loss.item()

                for i, th in enumerate(THRESHOLDS):
                    iou = compute_iou(gen_vol, gt_vol, th)
                    if i == 0:
                        IOU_40.append(iou)
                    elif i == 1:
                        IOU_50.append(iou)
                    elif i == 2:
                        IOU_75.append(iou)

            mean_loss = TEST_LOSS_ACCUMULATOR/(batch_idx+1)
            mean_IoU_40 = sum(IOU_40)/len(IOU_40)
            mean_IoU_50 = sum(IOU_50)/len(IOU_50)
            mean_IoU_75 = sum(IOU_75)/len(IOU_75)


    return mean_loss, (mean_IoU_40, mean_IoU_50, mean_IoU_75)






def initiate_training_environment(path: str):
    if not os.path.exists(path):
        os.mkdir(os.path.join(path))
    new_path = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, "weights"))
    os.mkdir(os.path.join(new_path, "samples"))

    return new_path


def train(configs):

    ## TODO: forward pass
    ## TODO: use modules gradually
    ## TODO: save weights every N epochs
    ## TODO: save weights when loss decreases under best loss
    ## TODO: log every [batch_idx condition]

    writer = Writer(configs["train_path"])
    train_cfg = configs["train"]
    augmentation_cfg = configs["augmentation"]
    model_cfg = configs["model"]
    dataset_cfg = configs["dataset"]
    THRESHOLDS = configs["thresholds"]
    BATCH_SIZE = train_cfg["batch_size"]
    
    scaler = torch.amp.GradScaler('cuda')
    train_transformations = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.RandomApply([T.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.2),
        T.Lambda(lambda x: torch.clamp(x, 0.0, 255.0)),
        T.ColorJitter(brightness=augmentation_cfg["brightness"],
                      contrast=augmentation_cfg["contrast"],
                      saturation=augmentation_cfg["saturation"]),
    ])
    test_transformations = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])

    train_dataset = ShapeNet3DDataset(dataset_path=dataset_cfg["data_path"],
                                      json_file_path=dataset_cfg["json_mapper"],
                                      split='train',
                                      transforms=train_transformations)
    train_dataset.set_n_views_rendering(1)
    train_dataset.choose_images_indices_for_epoch()

    test_dataset = ShapeNet3DDataset(dataset_path=dataset_cfg["data_path"],
                                      json_file_path=dataset_cfg["json_mapper"],
                                      split='test',
                                      transforms=test_transformations)
    test_dataset.set_n_views_rendering(1)
    test_dataset.choose_images_indices_for_epoch()

    

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # Modules
    Encoder = encoder.Encoder(configs=model_cfg).to(configs["device"])
    Decoder = decoder.Decoder().to(configs["device"])
    Merger = merger.Merger(model_cfg["lrelu_factor"]).to(configs["device"])
    Refiner = refiner.Refiner(model_cfg["lrelu_factor"], model_cfg["use_bias"]).to(configs["device"])

    # Weights initialization [meaningless since BN is used]
    Encoder.apply(network_utils.init_weights)
    Decoder.apply(network_utils.init_weights)
    Merger.apply(network_utils.init_weights)
    Refiner.apply(network_utils.init_weights)

    # Defining Optimizers
    learning_rate = train_cfg["lr"]
    E_optim = torch.optim.Adam(Encoder.parameters(), lr=learning_rate, weight_decay=1e-2)
    D_optim = torch.optim.Adam(Decoder.parameters(), lr=learning_rate, weight_decay=1e-3)
    M_optim = torch.optim.Adam(Merger.parameters(), lr=learning_rate, weight_decay=1e-3)
    R_optim = torch.optim.Adam(Refiner.parameters(), lr=learning_rate, weight_decay=1e-3)


    ## loss configs
    loss_fn = VoxelLoss(weight=10) # cuz loss is always multiplied by 10 in original p2v impl
    best_val_iou = 0

    BATCH_SIZE = train_cfg["batch_size"]
    n_views = 1
    USE_MERGER = train_cfg["epochs_till_merger"] == 0
    # USE_REFINER = train_cfg["epochs_till_refiner"] == 0
    EPOCHS = train_cfg["epochs"]
    ITERATIONS_PER_EPOCH = int(len(train_dataset)/BATCH_SIZE)
    START_EPOCH = train_cfg["start_epoch"]
    drawing_counter = 0
    for epoch in range(START_EPOCH, EPOCHS):
        LOG("EPOCH", epoch+1)
        if epoch == train_cfg["epochs_till_merger"]:
            writer.add_line("MERGER WILL NOW BE USED")
            USE_MERGER = True
        # if epoch == train_cfg["epochs_till_refiner"]:
            # writer.add_line("REFINER WILL NOW BE USED")
            # USE_REFINER = True
        
        
        Encoder.train()
        Decoder.train()
        Merger.train()
        Refiner.train()
        TRAIN_LOSS_ACCUMULATOR = 0
        LOG("TRAINING")
        with torch.amp.autocast("cuda"):

            for batch_idx, batch in enumerate(train_loader):
                E_optim.zero_grad()
                D_optim.zero_grad()
                M_optim.zero_grad()
                R_optim.zero_grad()
                v_img, r_img, gt_vol = batch
                v_img = v_img.to("cuda")
                r_img = r_img.to("cuda")
                gt_vol = gt_vol.to("cuda")
                
                #ENCODER
                lvl0, lvl1, lvl2, lvl3, latent_space = Encoder(v_img, r_img)
                
                #DECODER
                base_input = merge_feature_maps(BATCH_SIZE, n_views, lvl3, latent_space)
                raw, gen_vol = Decoder(lvl0, lvl1, lvl2, base_input)
                #MERGER
                if USE_MERGER:
                    gen_vol = Merger(raw, gen_vol)
                else:
                    gen_vol = gen_vol.squeeze(dim=1)
                #REFINER
                gen_vol = Refiner(gen_vol)

                gen_vol = gen_vol.squeeze(dim=1)

                loss = loss_fn(gen_vol, gt_vol)
                TRAIN_LOSS_ACCUMULATOR+=loss.item()

                scaler.scale(loss).backward()

                scaler.step(E_optim)
                scaler.step(D_optim)
                if USE_MERGER:
                    scaler.step(M_optim)

                # if USE_REFINER:
                scaler.step(R_optim)

                scaler.update()
                  ##TODO: DRAW SOME SHAPES EVERY WHILE
                if batch_idx % train_cfg["print_every"] == 0:
                    LOG(loss.item())

            mean_loss = TRAIN_LOSS_ACCUMULATOR/ITERATIONS_PER_EPOCH

        LOG("TESTING")
        valid_loss, valid_IoU = compute_validation_metrics(Encoder, Decoder, Merger, Refiner, test_loader, loss_fn, n_views, THRESHOLDS, USE_MERGER)
        LOG("average test loss", valid_loss)
        LOG("average test IoU", valid_IoU)
        mean_iou = sum(valid_IoU)/len(valid_IoU)
        LOG("average train loss", mean_iou)

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            drawing_counter+=1
            LOG(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")
            writer.add_line(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")

            weights_path = os.path.join(configs["train_path"], "weights", "best.pth")
            network_utils.save_checkpoints(weights_path, epoch+1, Encoder, E_optim, Decoder, D_optim, Refiner, R_optim , Merger,  M_optim, mean_iou, epoch+1)
            if drawing_counter == train_cfg["save_every"]:
                drawing_counter = 0
                samples_path = os.path.join(configs["train_path"], "samples", f"output{epoch+1}.pth")
                torch.save(gen_vol, samples_path)

            LOG("tensor saved")

        if (epoch+1) % train_cfg["save_every"] == 0:
            weights_path = os.path.join(configs["train_path"], "weights", "last.pth")
            network_utils.save_checkpoints(weights_path, epoch+1, Encoder, E_optim, Decoder, D_optim, Refiner, R_optim , Merger,  M_optim, mean_iou, epoch+1)

        if (epoch+1) % train_cfg["reduce_lr_epochs"]== 0:
            LOG("REDUCING LR")
            reduce_lr_factor = train_cfg["reduce_lr_factor"]
            learning_rate*= reduce_lr_factor
            writer.add_line(f"Learning rate has been reduced to {learning_rate} at epoch {epoch+1}")
            writer.add_line(f"")
            for param_group in E_optim.param_groups:
                param_group['lr'] *= reduce_lr_factor
            for param_group in D_optim.param_groups:
                param_group['lr'] *= reduce_lr_factor
            for param_group in M_optim.param_groups:
                param_group['lr'] *= reduce_lr_factor
            for param_group in R_optim.param_groups:
                param_group['lr'] *= reduce_lr_factor

        writer.add_scaler("TRAIN LOSS", epoch+1, mean_loss)
        writer.add_scaler("VALID LOSS", epoch+1, valid_loss)
        writer.add_scaler("VALID IoU@40", epoch+1, valid_IoU[0])
        writer.add_scaler("VALID IoU@50", epoch+1, valid_IoU[1])
        writer.add_scaler("VALID IoU@75", epoch+1, valid_IoU[2])
        writer.add_scaler("Mean IoU", epoch+1, mean_iou)

        n_views = update_dataset_configs(train_loader, USE_MERGER)
        test_loader.dataset.set_n_views_rendering(n_views)
        test_loader.dataset.choose_images_indices_for_epoch()

    writer.close()









def main():
    configs = None
    with open("/kaggle/working/3D-Reconstruction-GP/config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    DEBUGGER_SINGLETON.active = configs["use_debugger"]

    train_path = initiate_training_environment(configs["train"]["output_dir"])
    configs["train_path"] = train_path
#     ##TODO: CALL THESE 2 NIGGAS AT THE END OF EACH EPOCH
#     train_dataloader.dataset.set_n_views_rendering(10) ## 12 is a random number from lower bound 2 upper bound in config file
#     train_dataloader.dataset.choose_images_indices_for_epoch()
#     # train_pix2vox(cfg, train_dataloader, train_dataloader)
    train(configs=configs)



if __name__ == "__main__":
    main()






