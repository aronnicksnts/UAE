import pandas
import torch
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils import data

DATA_PATH = 'datasets'

class Xray(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(Xray, self).__init__()
        self.transform = transform
        self.file_path = []
        self.labels = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        for label in os.listdir(main_path):
            if label not in ['0', '1']:
                continue
            for file_name in tqdm(os.listdir(main_path+'/'+label)):
                data = sitk.ReadImage(main_path+'/'+label + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze()
                img = Image.fromarray(data).convert('L').resize((img_size,img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(int(label))
        

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.slices)


def get_xray_dataloader(bs, workers, dtype='train', img_size=64, dataset='dataset_1'):
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = f"{DATA_PATH}/{dataset}/"

    path += dtype
    
    dset = Xray(main_path=path, transform=transform, img_size=img_size)
    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader
