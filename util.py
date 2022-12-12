import os
import cv2
import numpy as np
import random
import torch
import logging

from tensorboardX import SummaryWriter

class Tools():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if 'train' in args.config:
            self.tensorboard_writer = self.build_tensorboard()

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

    def build_tensorboard(self):
        """Init tensorboard writer."""
        log_dir = os.path.join(self.config['path']['save_path'], 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)

    def plot(self, loss, iteration, title, lr=None):
        for loss_type, value in loss.items():
            self.tensorboard_writer.add_scalar("{}/{}".format(title, loss_type), value, iteration)
        if lr is not None:
            self.tensorboard_writer.add_scalar('Train/lr', lr, iteration)

    def save_model(self, model, optimizer, scheduler, iteration, eval_loss, filepath):
        state = {
            'iteration': iteration,
            'model_type': self.args.model,
            'batch_size': self.args.batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'eval_loss': eval_loss,
        }
        torch.save(state, filepath)

    # for training
    def save_img(self, output, fname, iteration):
        fname = fname.split('.')[0]
        output = output.cpu().detach().numpy()
        img = output[0]
        img *= 255
        img = img.astype(int)
        os.makedirs(os.path.join(self.config['path']['save_path'], 'predict'), exist_ok=True)
        file_path = os.path.join(self.config['path']['save_path'], 'predict/{}({}).png'.format(fname, iteration))
        cv2.imwrite(file_path, img)

    # for inference
    def inference_img(self, outputs, fnames):
        for i in range(len(fnames)):
            fname = fnames[i].split('/')[-1]
            output = outputs[i].cpu().detach().numpy()
            img = output[0]
            img *= 255
            img = img.astype(int)
            # save the image
            model_name = self.config['path']['load_path'].split('/')[-1]
            file_type = self.config['path']['dataset'].split('/')[-1]
            os.makedirs(os.path.join(self.config['path']['save_path'], '{}/{}'.format(model_name, file_type)), exist_ok=True)
            file_path = os.path.join(self.config['path']['save_path'], '{}/{}/{}'.format(model_name, file_type, fname))
            print(file_path, img)
            cv2.imwrite(file_path, img)

    # for inference 1
    def show_inference_img(self, outputs, fnames):
        fname = fnames[0].split('/')[-1]
        output = outputs[0].cpu().detach().numpy()
        img = output[0]
        img *= 255
        img = img.astype(np.uint8)
        cv2.imwrite('BSE_{}'.format(fname), img)
        cv2.imshow('BSE_{}'.format(fname), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_logger(self):
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        # file to log to
        logFile = os.path.join(self.config['path']['save_path'], 'log.txt')
        # setup File handler
        file_handler = logging.FileHandler(logFile)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        # setup Stream Handler (i.e. console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        stream_handler.setLevel(logging.INFO)
        # get our logger
        logger = logging.getLogger('root')
        logger.setLevel(logging.INFO)
        # add both Handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger