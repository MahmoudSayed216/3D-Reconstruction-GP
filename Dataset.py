import os
import torch
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
import json
from utils.utils import load_binvox_as_tensor
from PIL import Image
import torchvision.transforms as T
from utils.debugger import DEBUG

class ShapeNet3DDataset(Dataset):

    def __init__(self, dataset_path, json_file_path, split='train', transforms=None):
        self.classes = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
        self.dataset_path = dataset_path
        self.json_file_path = json_file_path
        self.split = split
        self.transform = transforms
        self.different_views_per_model = 24
        self.list_of_24 = list(range(24))
        with open(json_file_path, 'r') as file: json_file = json.load(file)

        self.data = []
        # for i in range(len(json_file)): self.data += json_file[i][split]
        
        for i in range(len(json_file)):
            current = json_file[i][split]
            for j, file in enumerate(current):
                file = self.classes[i] + file
                current[j] = file

            self.data.extend(current)


    def __len__(self):
        return len(self.data)
    
    
    ##TODO: CHANGE N VIEWS EACH EPOCH
    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def _choose_images_indices_for_epoch(self):
        self.random_indices = random.sample(self.list_of_24, self.n_views_rendering)
            


    def __getitem__(self, index):
        cls, current = self.data[index][0:8], self.data[index][8:]
        # cls = self.classes[class_idx]
        self._choose_images_indices_for_epoch()
        images_base = os.path.join(self.dataset_path, "ShapeNetRendering/ShapeNetRendering",cls, current)
        images_paths = sorted(os.listdir(images_base))
        chosen_images = [images_paths[i] for i in self.random_indices]
        model_path = os.path.join(self.dataset_path, "ShapeNetVox32/ShapeNetVox32", cls,current,"model.binvox")

        volume = load_binvox_as_tensor(model_path)
        v_images = []
        r_images = []
        for image in chosen_images:
            image_path = os.path.join(images_base, image)
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
            v_img = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
            r_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            v_images.append(v_img)
            r_images.append(r_img)
        v_images = torch.stack(v_images, dim=0)
        r_images = torch.stack(r_images, dim=0)
        return v_images, r_images, volume
