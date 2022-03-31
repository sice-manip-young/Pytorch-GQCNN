import torch
from torch.utils.data import Dataset

import os

from tqdm import tqdm
import numpy as np

# load the dexnet-2.0 dataset
# @return: image_tensor, depth_tensor, grasp_metric_tensor
def load_dexnet2(opt, s_data=0, n_data=1000):
    images  = []
    depth   = []
    gquality = []
    for i in tqdm(range(s_data, s_data+n_data)):
            index = str(i).zfill(5)
            path_depth  = os.path.join('data', opt.dataset_name, 'tensors', 'depth_ims_tf_table_'+str(index)+'.npz')
            path_hand_poses   = os.path.join('data', opt.dataset_name, 'tensors', 'hand_poses_'+str(index)+'.npz')
            path_metric = os.path.join('data', opt.dataset_name, 'tensors', 'robust_ferrari_canny_'+str(index)+'.npz') 

            depth_im_set = np.load(path_depth)
            hand_poses_set = np.load(path_hand_poses)
            grasp_metric_set = np.load(path_metric)

            images.append(depth_im_set)
            depth.append(hand_poses_set)
            gquality.append(grasp_metric_set)
    return images, depth, gquality

# @return: image_tensor, depth_tensor, grasp_metric_tensor
class ImageDataset(Dataset):
    def __init__(self, images, depth, gquality, transforms_=None):
        self.transform = transforms_

        self.images   = images
        self.depth    = depth
        self.gquality = gquality
    
    def __getitem__(self, index):

        img_group_id = int(index//1000)
        img_one_id = int(index%1000)

        image = self.images[img_group_id]['arr_0'][img_one_id,...]
        hand_pose = self.depth[img_group_id]['arr_0'][img_one_id,...]
        grasp_metric = self.gquality[img_group_id]['arr_0'][img_one_id,...]
        
        # depth image
        image = image.astype(np.float32)
        image = self.transform(image)

        # gripper depth
        depth = hand_pose[2]
        depth = depth.astype(np.float32)

        # grasp quality
        grasp_metric =  grasp_metric.astype(np.float32)  

        return image, depth, grasp_metric 

    def __len__(self):
        return len(self.images)*1000
