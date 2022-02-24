import numpy as np

import os
import glob
import argparse

import matplotlib.pyplot as plt

from tqdm import tqdm

'''
 各npzファイルには1000個分のデータが入っている
 1) depth_ims_tf_tabel
    32 x 32 x 1 - {height x width x channel}
 2) hand_poses
    0: row index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
    1: column index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
    2: depth, in meters, of gripper center from the camera that took the corresponding depth image
	3: angle, in radians, of the grasp axis from the image x-axis (middle row of pixels, pointing right in image space)
	4: row index, in pixels, of the object center projected into a depth image centered on the world origin
	5: column index, in pixels, of the object center projected into a depth image centered on the world origin
	6: width, in pixels, of the gripper projected into the depth image
 3) robust_ferrari_canny
 4) ferrari_canny
 5) force_closure
'''

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="dex-net_2.0", help="name of the dataset")
    opt = parser.parse_args()
    
    
    # datasets = glob.glob(os.path.join('data', opt.dataset_name, '**', '*'))
    # print (len(datasets))
    # basename = os.path.basename(datum)
    # lastname = basename.split('_')
    # index = lastname[-1].split('.') [0]

    for i in tqdm(range(10)):
        index = str(i).zfill(5)
        path_depth  = os.path.join('data', opt.dataset_name, 'tensors', 'depth_ims_tf_table_'+str(index)+'.npz')
        path_hand_poses   = os.path.join('data', opt.dataset_name, 'tensors', 'hand_poses_'+str(index)+'.npz')
        path_metric = os.path.join('data', opt.dataset_name, 'tensors', 'robust_ferrari_canny_'+str(index)+'.npz') 

        depth_im = np.load(path_depth)
        hand_poses = np.load(path_hand_poses)
        grasp_metric = np.load(path_metric)

        # print ('depth: ', depth_im['arr_0'][478,...])
        # print ('hand_poses: ', hand_poses['arr_0'][478,...])
        # print ('grasp_metric: ', grasp_metric['arr_0'][478,...])
        
        plt.imsave('./tools/images/depth_%s.png' % (index), depth_im['arr_0'][0,:,:,0], cmap='gray')
