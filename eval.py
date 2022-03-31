from email.policy import strict
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.options import options
from utils.dataloader import load_dexnet2, ImageDataset

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

    images_valid, depth_valid , gquality_valid = load_dexnet2(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset)
    image_dataset = ImageDataset(images_valid, depth_valid , gquality_valid, transforms_=data_transforms['val'])
    
    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    path_model = "saved_models/%s/model_latest.pth" % (opt.dataset_name)
    net.load_state_dict(torch.load(path_model), strict=False)
    print ('load {}'.format(path_model))
    
    # save the result on the text file
    f = open('sample_demo/%s/result.csv' % (opt.dataset_name), 'w')

    data_corrects = 0
    net.eval()
    for i, (images, z, gq) in enumerate(dataloader):
        gq = gq.view(gq.size()[0], -1)
        grasp = torch.where(gq>opt.gamma, 1, 0)
        grasp = torch.squeeze(grasp, dim=1)

        outputs = net (images, z)

        if opt.save_image_eval:
            result = images[0,0].numpy()
            fig = plt.figure()
            plt.imshow(result); 
            plt.axis('off'); 
            plt.tight_layout()    
            fig.savefig('sample_demo/%s/%s' % (opt.dataset_name, str(i)))
            # release memory
            plt.clf()
            plt.close()

        correct = (grasp == torch.max(outputs, 1)[1])

        data_corrects += torch.sum(correct)

        prob = torch.nn.Softmax(dim=1)(outputs)
        print ('Id: %d, prob: %.5f, GQ: %.5f, T/F: %s'% (i, prob[0,1].item(), gq.item(), correct.item()))
        f.write('%d, %.8f, %.8f, %s\n'% (i, prob[0,1].item(), gq.item(), correct.item()))
        

        if i >= opt.n_sample-1:
            break

    f.close()

    print ('Accuracy:', ((data_corrects/opt.n_sample).item() * 100), '[%]')
