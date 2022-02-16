import argparse
import os

import torch

from utils.options import options
# from utils.dataloader import dataloader
from gqcnn.gqcnn import gqcnn

if __name__=='__main__':

    # --- cuda
    cuda = True if torch.cuda.is_available() else False

    # --- config netowrk 
    opt = options()
    print (opt)

    gqcnn = gqcnn(im_size=128)
    print (gqcnn)

    # criterion

    # train
     
    # eval