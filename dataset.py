import os
import glob
import cv2
import torch
import json
import numpy as np
import torchvision.utils as utils
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
from random import sample

class JSRT_Dataset(Dataset):
    def __init__(self, src, tgt=None, data_aug_list=[], resize=256, pca_source=None) -> None:
        super().__init__()
        self.data_aug_list = data_aug_list
        self.source = src
        self.target = tgt
        self.transform = self.get_fn(resize)
        self.pca_source = pca_source

    def __getitem__(self, idx):
        # source data
        src = self.source[idx]
        fname = src.split('/')[-1]
        if self.pca_source:
            src_img = torch.tensor(self.pca_source[idx], dtype=torch.float32)
        else:
            src_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if 'equalhist' in self.data_aug_list:
                src_img = cv2.equalizeHist(src_img)
            src_img = src_img / 255
            src_img = Image.fromarray(src_img)
            src_img = self.transform(src_img)
        # target data
        if self.target is not None:
            tgt = self.target[idx]
            tgt_img = cv2.imread(tgt, cv2.IMREAD_GRAYSCALE)
            tgt_img = tgt_img / 255
            tgt_img = Image.fromarray(tgt_img)
            # 256
            tgt_img = self.transform(tgt_img)
            # 1024
            # toTensor = transforms.ToTensor()
            # tgt_img = toTensor(tgt_img)
            return src_img, tgt_img, fname
        else:
            return src_img, fname


    def __len__(self):
        return len(self.source)

    def get_fn(self, resize):
        compose = []
        compose.append(transforms.Resize((resize, resize)))
        compose.append(transforms.ToTensor())
        transform = transforms.Compose(compose)
        return transform


class BS_Dataloader(DataLoader):
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

    def get_dataloader(self):
        src = sorted(glob.glob(os.path.join(self.config['path']['dataset'], 'source/*')))
        tgt = sorted(glob.glob(os.path.join(self.config['path']['dataset'], 'target/*')))
        if self.args.model == 'Decoder':
            src_pca = self.get_pca_data()
            train_src_pca = []
            val_src_pca   = []
        # random split to training and validation set
        num_of_train = int(len(src) * (1 - self.config['parameter']['train_set_ratio']))
        samp = sample(range(len(src)), num_of_train)
        train_src = []
        train_tgt = []
        val_src   = []
        val_tgt   = []
        for i in range(len(src)):
            if i in samp:
                if self.args.model == 'Decoder':
                    train_src_pca.append(src_pca[i])
                train_src.append(src[i])
                train_tgt.append(tgt[i])
            else:
                if self.args.model == 'Decoder':
                    val_src_pca.append(src_pca[i])
                val_src.append(src[i])
                val_tgt.append(tgt[i])
        # get dataset
        if self.args.model == 'Decoder':
            src_pca = self.get_pca_data()
            train_set = JSRT_Dataset(train_src, train_tgt, resize=self.config['parameter']['img_resize'], pca_source=train_src_pca)
            val_set   = JSRT_Dataset(val_src, val_tgt, resize=self.config['parameter']['img_resize'], pca_source=val_src_pca)
        else:
            train_set = JSRT_Dataset(train_src, train_tgt, resize=self.config['parameter']['img_resize'])
            val_set   = JSRT_Dataset(val_src, val_tgt, resize=self.config['parameter']['img_resize'])
        print('train set: ', len(train_set))
        print('val set: ', len(val_set))
        # get dataloader
        if self.args.debug:
            print('debug mode')
            train_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_set  , batch_size=self.args.batch_size, shuffle=False)
        return train_loader, val_loader


    def get_inference_dataloader(self, src=None):
        if src is None:
            print(os.path.join(self.config['path']['dataset'], 'image/*'))
            srcs = sorted(glob.glob(os.path.join(self.config['path']['dataset'], 'image/*')))
            dataset = JSRT_Dataset(srcs, data_aug_list=['equalhist'])
        else:
            print([src])
            if self.args.model == 'Decoder':
                dataset = JSRT_Dataset([src], data_aug_list=['PCA'])
            else:
                dataset = JSRT_Dataset([src])
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return dataloader

    def get_pca_data(self):
        with open('./dataset.json') as file:
            data = json.load(file)
            return data['pca_source']