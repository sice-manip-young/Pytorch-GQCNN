import torch.nn.functional as F
import torch.nn as nn
import torch

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
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=(7//2, 7//2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(5//2, 5//2)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(64), 

            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(3//2, 3//2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(3//2, 3//2)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(64),
        )

        self.image_fc = nn.Sequential(
            nn.Linear(in_features=64 * im_size//2 * im_size//2, out_features=1024),  #TODO
            nn.ReLU(inplace=True),
        )
        
        self.z_fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=16), 
            nn.ReLU(inplace=True),
        )

        self.union_models = nn.Sequential(
            nn.Linear(in_features=(1024+16), out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2),
        )

        self.activate = nn.Softmax(dim=1)

    def forward(self, y, z):
        y = self.image_conv(y)
        y = y.view(y.size()[0], -1)
        y = self.image_fc(y)

        z = z.view(z.size()[0], -1)
        z = self.z_fc(z)

        # concat
        output = torch.cat ([y, z], dim=1)
        output = self.union_models(output)
        q_theta = self.activate(output)
        
        return q_theta.view(q_theta.size()[0], -1)