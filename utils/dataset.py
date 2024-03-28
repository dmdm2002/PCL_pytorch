import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random
import numpy as np


class AugDataset(data.Dataset):
    def __init__(self, cfg, base_trans, aug_trans) -> None:
        super().__init__()

        self.cfg = cfg

        self.fake_imgs = glob.glob(f"{os.path.join(self.cfg['dataset_path'], self.cfg['label'][0])}/*")
        self.live_imgs = glob.glob(f"{os.path.join(self.cfg['dataset_path'], self.cfg['label'][1])}/*")

        self.base_trans = base_trans
        self.aug_trans = aug_trans

        self.all_fake_imgs = []
        self.all_live_imgs = []
    
    def __len__(self):
        return len(self.fake_imgs)
    
    def __getitem__(self, idx):
        self.fake_img = self.base_trans(Image.open(self.fake_imgs[idx]))
        self.all_fake_imgs.append(self.fake_img)

        for _ in range(self.cfg['n_aug_imgs']-1):
            self.all_fake_imgs.append(self.aug_trans(self.fake_img))  # fake augmentation image를 생성

        self.live_img = self.base_trans(Image.open(self.live_imgs[idx]))
        self.all_live_imgs.append(self.live_img)

        for _ in range(self.cfg['n_aug_imgs']-1):
            self.all_live_imgs.append(self.aug_trans(self.live_img))  # live augmentation image를 생성
        
        return self.all_fake_imgs, self.all_live_imgs
