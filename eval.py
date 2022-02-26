import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.options import options
from utils.dataloader import ImageDatasetEval

from gqcnn.gqcnn import gqcnn

from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=='__main__':

    # config 
    opt = options()
    print (opt)

    # cuda
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs("sample_demo/%s" % (opt.dataset_name), exist_ok=True)

    net = gqcnn(im_size=32)
    # print (net)

    # data loader
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
    ])

    # image_dataset = ImageDatasetEval(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset, transforms_=data_transform)
    image_dataset = ImageDatasetEval(opt, s_data=5000, n_data=1, transforms_=data_transform)
    
    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    path_model = "saved_models/%s/model_latest.pth" % (opt.dataset_name)
    net.load_state_dict(torch.load(path_model))
    print ('load {}'.format(path_model))


    net.eval()
    with torch.no_grad():
        for i, (images, z, gq, poses) in enumerate(dataloader):
            outputs = net (images, z)

            result = images[0,0].numpy()
            fig = plt.figure()
            plt.imshow(result); 
            plt.axis('off'); 
            plt.tight_layout()
            
            fig.savefig('sample_demo/%s/%s' % (opt.dataset_name, str(i)))
            #fig.savefig('sample_demo/%s/%s_%s_%s_%s' % (opt.dataset_name, str(i), z.item(), gq.item(), outputs.item()))

            # print ('depth: ', z.item(), 'hand_poses: ', poses[0].numpy(), 'grasp_metric: ', gq.item(), )
            print (z.item(), gq.item(), outputs[0,0].numpy())

            if i > 10:
                break
