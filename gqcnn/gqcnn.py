import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class gqcnn (nn.Module):
    def __init__(self, im_size):
        super(gqcnn, self).__init__()

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=(7//2, 7//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(5//2, 5//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LocalResponseNorm(size=64, alpha=2.0e-05, beta=0.75, k=1.0),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3//2, 3//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3//2, 3//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LocalResponseNorm(size=64, alpha=2.0e-05, beta=0.75, k =1.0), 
        )

        self.image_fc = nn.Sequential(
            nn.Linear(in_features=64 * im_size//2 * im_size//2, out_features=1024),  #TODO
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.z_fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=16), 
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.merge_models = nn.Sequential(
            nn.Linear(in_features=(1024+16), out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=2),
        )

    def forward(self, y, z):
        y = self.image_conv(y)
        y = torch.flatten(y, 1)
        y = self.image_fc(y)

        z = z.view(z.size()[0], -1) # depth
        z = self.z_fc(z)

        # concat
        output = torch.cat ([y, z], dim=1)
        output = self.merge_models(output)
        
        return output.view(output.size()[0], -1)

class gqcnn_with_attention(nn.Module):
    def __init__(self, im_size):
        super(gqcnn_with_attention, self).__init__()

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=(7//2, 7//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(5//2, 5//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LocalResponseNorm(size=64, alpha=2.0e-05, beta=0.75, k=1.0),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3//2, 3//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3//2, 3//2), padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LocalResponseNorm(size=64, alpha=2.0e-05, beta=0.75, k =1.0), 
        )

        self.image_fc = nn.Sequential(
            nn.Linear(in_features=64,  out_features=1024),  #TODO
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.z_fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=16), 
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.merge_models = nn.Sequential(
            nn.Linear(in_features=(1024+16), out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=2),
        )

    def forward(self, y, z):
        # image features
        y = self.image_conv(y)

        # attention
        attn = self.attn_conv(y) # (B, 1, H, W)
        B, _, H, W = attn.shape
        self.mask = attn.detach().cpu() # save the attention mask 

        y = y * attn
        y = torch.nn.functional.adaptive_avg_pool2d(y, (1,1))
        y = torch.flatten(y, 1)
        y = self.image_fc(y)

        z = z.view(z.size()[0], -1) # depth
        z = self.z_fc(z)

        # concat
        output = torch.cat ([y, z], dim=1)
        output = self.merge_models(output)
        
        return output.view(output.size()[0], -1)

    def save_attention_mask(self, y, z, dir):
        batch_size = y.shape[0]
        self.forward(y, z)

        y = y.cpu()

        _, axs = plt.subplots(1,2, figsize=(6,3))
        # plt.axis('off')
        axs[0].imshow(y[0].permute(1,2,0), vmin=0., vmax=1.)
        axs[1].imshow(self.mask[0][0], cmap='viridis', vmin=0., vmax=1.)
        plt.savefig(dir)
        plt.close()
        return 