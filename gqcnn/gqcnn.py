import torch.nn.functional as F
import torch.nn as nn
import torch


class gqcnn (nn.Module):
    def __init__(self):
        super(gqcnn, self).__init__()

        # definite models
        self.model = self.gqcnn_network_model()

    def gqcnn_network_model(self):
        # 
        conv1 = nn.Sequential()

        #
        conv2 = nn.Sequential()

        #
        fc = nn.Sequential()

        #

        model = nn.Sequential()
        
        return model

    def forward(self, x):
        return self.model(x)