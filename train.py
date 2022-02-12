import argparse
import os

import torch

from utils import options, dataloader
from gqcnn import gqcnn

if __name__=='__main__':

    # --- cuda
    cuda = True if torch.cuda.is_available() else False

    # --- config netowrk 
    opt = options()
    print (opt)

    gqcnn = gqcnn()

    # criterion



    # train
     
    # eval