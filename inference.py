import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torchvision
import platform

from argparse import ArgumentParser
from tqdm import tqdm
from dataset import BS_Dataloader
from network import Model
from util import Tools

def parse():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/inference.yaml', type=str, help='path of config file')
    parser.add_argument('--model', default='Resnet_BS', help=['model architecture for bone suppression'])
    parser.add_argument('--batch_size', default=2, help='batch size')
    parser.add_argument('--inference_src', default=None, help='inference on assign image')
    args = parser.parse_args()
    return args

def inference(args, tools, dataloader, model):
    model.eval()
    device = model.device
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='[inference]'):
            src, pca, fnames = batch
            src = src.to(device)
            outputs = model(src)
            if args.inference_src is None:
                tools.inference_img(outputs, fnames)
            else:
                tools.show_inference_img(outputs, fnames)


def main():
    args = parse()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    os.makedirs(config['path']['save_path'], exist_ok=True)
    # identify OS
    if platform.system() == 'Darwin':
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get tool
    tools = Tools(args, config)
    tools.set_seed(328)
    # get dataloader
    bs_dataloader = BS_Dataloader(args, config)
    dataloader = bs_dataloader.get_inference_dataloader(src=args.inference_src)
    # get model
    model_class = Model(args, config, device)
    model, _ = model_class.get_save_model(os.path.join(config['path']['load_path'], 'best.pth'))
    # start train
    inference(args, tools, dataloader, model)

if __name__ == '__main__':
    main()