import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torchvision
import platform
# import tensorflow as tf

from argparse import ArgumentParser
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from sklearn.metrics import mean_absolute_error
from dataset import BS_Dataloader
from network import Model
from util import Tools
from metric import MSSSIM

def parse():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/train.yaml', type=str, help='path of config file')
    parser.add_argument('--model', default='Resnet_BS', help=['model architecture for bone suppression'])
    parser.add_argument('--batch_size', default=8, help='batch size')
    parser.add_argument('--optimizer', default='Adam', help='optimizer for training')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='scheduler for training')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    parser.add_argument('--val_freq', default=10000, help='num of iteration per evaluation')
    parser.add_argument('--debug', action='store_true', help='use smaller dataset and try to overfit')
    args = parser.parse_args()
    return args

def computing_loss(source, target):
    l1 = nn.L1Loss()
    mae_loss = l1(source, target)

    # jorge-pessoa pytorch version
    ms_ssim_module = MSSSIM(window_size=11, size_average=True, channel=1, normalize='relu')
    ms_ssim_loss = 1 - ms_ssim_module(source, target)

    '''
    # VainF pytorch version
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1)

    # tensorflow version
    source, target = source.cpu().detach().numpy(), target.cpu().detach().numpy()
    source = tf.convert_to_tensor(source, dtype=tf.float32)
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    ms_ssim_loss = 1-tf.reduce_mean(tf.image.ssim_multiscale(target, source, 1.0))
    '''

    combined_loss = 0.16 * mae_loss + 0.84 * ms_ssim_loss
    return {'mae_loss': mae_loss, 'ms_ssim_loss': ms_ssim_loss, 'combined_loss': combined_loss}

def eval(args, data_loader, model):
    model.eval()
    device = model.device
    mae_loss = 0
    ms_ssim_loss = 0
    combined_loss = 0
    num_of_images = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='[Val]'):
            src, tgt, fnames = batch
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src)
            batch_loss = computing_loss(outputs, tgt)
            mae_loss += batch_loss['mae_loss'] * len(fnames)
            ms_ssim_loss += batch_loss['ms_ssim_loss'] * len(fnames)
            combined_loss += batch_loss['combined_loss'] * len(fnames)
            num_of_images += len(fnames)
    mae_loss /= num_of_images
    ms_ssim_loss /= num_of_images
    combined_loss /= num_of_images
    eval_loss = {'mae_loss': mae_loss, 'ms_ssim_loss': ms_ssim_loss, 'combined_loss': combined_loss}
    return eval_loss, outputs, fnames


def train(args, config, tools, train_loader, val_loader, model, start_iteration):
    logger = tools.get_logger()
    logger.info(args)
    # optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    # scheduler
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=0.00001, patience=10)

    device = model.device
    best_results = {'mae_loss': 100, 'ms_ssim_loss': 100, 'combined_loss': 100}
    iteration = start_iteration


    model.train()
    for epoch in range(config['parameter']['epoch']):
        for batch in tqdm(train_loader, desc='[train] epoch-{}'.format(epoch)):
            iteration += 1
            src, tgt, fnames = batch
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad() 
            outputs = model(src)
            batch_loss = computing_loss(outputs, tgt)
            batch_loss['combined_loss'].backward() 
            optimizer.step()

            # plot tensorboard
            lr = optimizer.param_groups[0]['lr']
            tools.plot(batch_loss, iteration, 'Train', lr=lr)

            # evaluation
            if  iteration==1 or iteration % args.val_freq == 0:
                eval_loss, outputs, fnames = eval(args, val_loader, model)
                tools.plot(eval_loss, iteration, 'Val')
                if eval_loss['combined_loss'] < best_results['combined_loss']:
                    best_results = eval_loss
                    filepath = os.path.join(config['path']['save_path'], 'best.pth')
                    tools.save_model(model, optimizer, scheduler, iteration, eval_loss, filepath)
                    logger.info(f"Update best model: iteration {iteration}")
                # show loss information
                logger.info(f"[ Train | {iteration} ] Combined loss = {batch_loss['combined_loss']:.4f}, MAE loss = {batch_loss['mae_loss']:.4f}, MS-SSIM loss = {batch_loss['ms_ssim_loss']:.4f}")
                logger.info(f"[ Val | {iteration} ] Combined loss = {eval_loss['combined_loss']:.4f}, MAE loss = {eval_loss['mae_loss']:.4f}, MS-SSIM loss = {eval_loss['ms_ssim_loss']:.4f}")
                # save example image
                tools.save_img(outputs[-1], fnames[-1], iteration)
                model.train()

        # scheduler
        scheduler.step(eval_loss['combined_loss'])

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
    train_loader, val_loader = bs_dataloader.get_dataloader()
    # get model
    model_class = Model(args, config, device)
    if not config['path']['load_path']:
        model = model_class.get_model()
        start_iteration = 0
    else:
        model, start_iteration = model_class.get_save_model(os.path.join(config['path']['load_path'], 'best.pth'))
    # start train
    train(args, config, tools, train_loader, val_loader, model, start_iteration)


if __name__ == '__main__':
    main()