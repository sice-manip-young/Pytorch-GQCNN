# Overview:Pytorch-GQCNN

Our team is portig the GQ-CNN with Dex-Net to Pytorch with reference to https://berkeleyautomation.github.io/gqcnn/ [1]. 

## Download

1. Download the dex-net 2.0 [2] directly. 
~~~
$ sh ./scripts/download_dexnet_2.sh
~~~  
2. Download the pretrained model for Tensorflow. **Note that ours does not support its model.**
~~~
$ sh ./scripts/download_model_dexnet_2.sh
~~~

## Running Locally
  
#### Train
~~~
$ python train.py --lr 0.01 --batch_size 128 --n_iterations 10000
~~~
#### Evaluation
~~~
$ python eval.py 
~~~

## Reference
[1] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg, “Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics,” arXiv preprint arXiv:1703.09312, 2017.

[2] Mahler, J. Releasing the Dexterity Network (Dex-Net) 2.0 Dataset for Deep Grasping. Available online: http://bair.berkeley.edu/blog/2017/06/27/dexnet-2.0/. (accessed on 1 April 2022).
