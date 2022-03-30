from email.policy import strict
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.options import options
from utils.dataloader import ImageDatasetEval, load_dexnet2, ImageDataset

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

    # data loader
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.001, 0.005) ), 
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    }

    # images_valid, depth_valid , gquality_valid = load_dexnet2(opt, s_data=0, n_data=opt.n_dataset)
    images_valid, depth_valid , gquality_valid = load_dexnet2(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset)
    # image_dataset = ImageDatasetEval(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset, transforms_=data_transform)
    # image_dataset = ImageDatasetEval(opt, s_data=1900, n_data=1, transforms_=data_transform)
    image_dataset = ImageDataset(opt, images_valid, depth_valid ,gquality_valid, s_data=opt.n_dataset, n_data=opt.n_valid_dataset, transforms_=data_transforms['val'])
    
    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    path_model = "saved_models/%s/model_latest.pth" % (opt.dataset_name)
    net.load_state_dict(torch.load(path_model), strict=False)
    print ('load {}'.format(path_model))

    gamma = opt.gamma
    data_corrects = 0

    net.eval()
    # with torch.no_grad():
    for i, (images, z, gq) in enumerate(dataloader):
        # images, z, gq = images.to(device), z.to(device), gq.to(device)

        gq = gq.view(gq.size()[0], -1)
        grasp = torch.where(gq>gamma, 1, 0)
        grasp = torch.squeeze(grasp, dim=1)

        outputs = net (images, z)

        # result = images[0,0].numpy()
        # fig = plt.figure()
        # plt.imshow(result); 
        # plt.axis('off'); 
        # plt.tight_layout()
        
        # fig.savefig('sample_demo/%s/%s' % (opt.dataset_name, str(i)))

        prob = torch.nn.Softmax(dim=1)(outputs)

        # print ('depth: ', z.item(), 'hand_poses: ', poses[0].numpy(), 'grasp_metric: ', gq.item(), )
        print (i, 'output: ', prob, 'gq: ', gq.item(),)

        data_corrects += torch.sum(grasp == torch.max(outputs, 1)[1])

        if i > 1000:
            break

    print ('Acc=', data_corrects/1000)
