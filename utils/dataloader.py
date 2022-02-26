import torch
from torch.utils.data import Dataset

import os

from tqdm import tqdm
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, opt, s_data=0, n_data=1000, transforms_=None):
        self.transform = transforms_

        self.images   = []
        self.depth    = []
        self.gqualiy = []
        self.len = 0

        for i in tqdm(range(s_data, s_data+n_data)):
            index = str(i).zfill(5)
            path_depth  = os.path.join('data', opt.dataset_name, 'tensors', 'depth_ims_tf_table_'+str(index)+'.npz')
            path_hand_poses   = os.path.join('data', opt.dataset_name, 'tensors', 'hand_poses_'+str(index)+'.npz')
            path_metric = os.path.join('data', opt.dataset_name, 'tensors', 'robust_ferrari_canny_'+str(index)+'.npz') 

            depth_im_set = np.load(path_depth)
            hand_poses_set = np.load(path_hand_poses)
            grasp_metric_set = np.load(path_metric)

            self.images.append(depth_im_set)
            self.depth.append(hand_poses_set)
            self.gqualiy.append(grasp_metric_set)

            self.len += depth_im_set['arr_0'].shape[0]

    def __getitem__(self, index):

        img_group_id = int(index/1000)
        img_one_id = index%1000

        image = self.images[img_group_id]['arr_0'][img_one_id,...]
        hand_pose = self.depth[img_group_id]['arr_0'][img_one_id,...]
        grasp_metric = self.gqualiy[img_group_id]['arr_0'][img_one_id,...]
        
        # depth image
        image = image.astype(np.float32)
        # image_tensor = torch.from_numpy(image).clone()
        image = self.transform(image)

        # gripper depth
        depth = hand_pose[2]
        depth = depth.astype(np.float32)
        # depth_tensor = torch.from_numpy(depth).clone()

        # grasp quality
        grasp_metric =  grasp_metric.astype(np.float32)
        #grasp_metric_tensor = torch.from_numpy(grasp_metric).clone()        

        return image, depth, grasp_metric #image_tensor, depth_tensor, grasp_metric_tensor

    def __len__(self):
        return self.len

class ImageDatasetEval(Dataset):
    def __init__(self, opt, s_data=0, n_data=1000, transforms_=None):
        self.transform = transforms_

        self.images   = []
        self.depth    = []
        self.gqualiy = []
        self.len = 0

        for i in range(s_data, s_data+n_data):
            index = str(i).zfill(5)
            path_depth  = os.path.join('data', opt.dataset_name, 'tensors', 'depth_ims_tf_table_'+str(index)+'.npz')
            path_hand_poses   = os.path.join('data', opt.dataset_name, 'tensors', 'hand_poses_'+str(index)+'.npz')
            path_metric = os.path.join('data', opt.dataset_name, 'tensors', 'robust_ferrari_canny_'+str(index)+'.npz') 

            depth_im_set = np.load(path_depth)
            hand_poses_set = np.load(path_hand_poses)
            grasp_metric_set = np.load(path_metric)

            self.images.append(depth_im_set)
            self.depth.append(hand_poses_set)
            self.gqualiy.append(grasp_metric_set)

            self.len += depth_im_set['arr_0'].shape[0]

    def __getitem__(self, index):

        img_group_id = int(index/1000)
        img_one_id = index%1000

        image = self.images[img_group_id]['arr_0'][img_one_id,...]
        hand_pose = self.depth[img_group_id]['arr_0'][img_one_id,...]
        grasp_metric = self.gqualiy[img_group_id]['arr_0'][img_one_id,...]
        
        # depth image
        image = image.astype(np.float32)
        image = self.transform(image)

        # gripper depth
        depth = hand_pose[2]
        depth = depth.astype(np.float32)

        # gripper pose
        hand_pose = hand_pose.astype(np.float32)

        # grasp quality
        grasp_metric =  grasp_metric.astype(np.float32)

        return image, depth, grasp_metric, hand_pose,

    def __len__(self):
        return self.len


