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
        elif self.args.model == 'AE_BS':
            model = AE_BS()
        elif self.args.model == 'Decoder':
            model = Decoder(in_dim=256)
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
        self.conv_final = nn.Conv2d(num_filters, 1, kernel_size=(3,3), padding='same')
        # 1024
        # self.upsample = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=(3,3), stride=2, padding=1)
        # self.conv_final = nn.ConvTranspose2d(num_filters, 1, kernel_size=(3,3), stride=2, padding=1)
        self.decoder = Decoder(in_dim=256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in, pca):
        x_in = self.conv_in(x_in)
        x = x_in.clone()
        for i in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[i](x)
        x = self.conv_out(x)
        output = x + x_in + self.decoder(pca)
        # 256
        output = self.conv_final(output)
        # assert 1 == 2
        # 1024
        # output = self.upsample(output, output_size=(512, 512))
        # output = self.conv_final(output, output_size=(1024, 1024))
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


class AE_BS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1, stride=2)
        # decoder
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), padding='same')
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(3,3), padding='same')
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=(3,3), padding='same')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()        

    def forward(self, x_in):
        # encoder
        x_en = self.conv1(x_in)
        x_en = self.relu(x_en)
        x_en = self.conv2(x_en)
        x_en = self.relu(x_en)
        x_en = self.conv3(x_en)
        x_en = self.relu(x_en)
        # decoder
        x_de = self.conv4(x_en)
        x_de = self.relu(x_de)
        x_de = self.up1(x_de, output_size=(128, 128))
        x_de = self.conv5(x_de)
        x_de = self.relu(x_de)
        x_de = self.up2(x_de, output_size=(256, 256))
        output = self.conv6(x_de)
        return self.sigmoid(output)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Decoder(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 1, 256, 256)
    """
    def __init__(self, in_dim, dim=256):
        super(Decoder, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 2 * 16 * 16, bias=False),
            nn.BatchNorm1d(dim * 2 * 16 * 16),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 2, dim * 1),
            dconv_bn_relu(dim * 1, dim // 2),
            dconv_bn_relu(dim // 2, dim // 4),
            nn.ConvTranspose2d(dim // 4, 1, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 16, 16)
        y = self.l2_5(y)
        print(y.shape)
        return y