from PIL import ImageFilter
import PIL.Image as Image
import random
import glob
import os
import torch.utils.data as data
import torchvision.datasets as datasets


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index
    

class CustomDataset(data.Dataset):
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
