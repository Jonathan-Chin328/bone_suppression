import torch
import torch.nn as nn

class Model():
    def __init__(self, args, config, device) -> None:
        self.args = args
        self.config = config
        self.device = device

    def get_model(self):
        if self.args.model == 'Resnet_BS':
            model = ResNet_BS(
                num_filters=self.config['model']['num_filters'], 
                num_res_blocks=self.config['model']['num_res_blocks'],
                res_block_scaling=self.config['model']['res_block_scaling'])
        else:
            assert('can not find available model')

        model = model.to(self.device)
        model.device = self.device
        return model

    def get_save_model(self, load_path):
        print('loading saved model...')
        ckpt = torch.load(load_path)
        print('=== Basic info ===')
        print('\tmodel type:', ckpt['model_type'])
        print('\titeration:', ckpt['iteration'])
        print('=== Performance (Combined loss, MAE loss, MS-SSIM loss) ===')
        print('\tCombined loss:', ckpt['eval_loss']['combined_loss'])
        print('\tMAE loss:', ckpt['eval_loss']['mae_loss'])
        print('\tSSIM loss:', ckpt['eval_loss']['ms_ssim_loss'])
        model = self.get_model()
        model.load_state_dict(ckpt['model_state_dict'])
        return model, ckpt['iteration']


class ResNet_BS(nn.Module):
    def __init__(self, num_filters=64, num_res_blocks=16, res_block_scaling=0.1) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(1, num_filters, kernel_size=(3,3), padding='same')
        self.resnet_blocks = nn.ModuleList([ResNet_Block(num_filters, res_block_scaling) for i in range(num_res_blocks)])
        self.conv_out = nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), padding='same')
        # 256
        # self.conv_final = nn.Conv2d(num_filters, 1, kernel_size=(3,3), padding='same')
        # 1024
        self.upsample = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=(3,3), stride=2, padding=1)
        self.conv_final = nn.ConvTranspose2d(num_filters, 1, kernel_size=(3,3), stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x_in = self.conv_in(x_in)
        x = x_in.clone()
        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)
        x = self.conv_out(x)
        output = x + x_in
        # 256
        # output = self.conv_final(output)
        # 1024
        output = self.upsample(output, output_size=(512, 512))
        output = self.conv_final(output, output_size=(1024, 1024))
        return self.sigmoid(output)


class ResNet_Block(nn.Module):
    def __init__(self, num_filters, scaling=None) -> None:
        super().__init__()
        self.scaling = scaling
        self.conv_1 = nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), padding='same')
        self.conv_2 = nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), padding='same')
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.conv_1(x_in)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.scaling is not None:
            x *= self.scaling
        return x + x_in