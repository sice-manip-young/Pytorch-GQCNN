# Overview:Pytorch-GQCNN
  ### Porting

# Environment

# Download
  ### Download the dex-net 2.0 (dataset)
  $ sh ./scripts/download_dexnet_2.sh
  
  ### Download the pretrained models
  $ sh ./scripts/download_model_dexnet_2.sh
  
# Train

  ### Example
  $ python train.py --n_dataset 1000 --n_valid_dataset 100 --batch_size 128
