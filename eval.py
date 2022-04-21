import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.options import options
from utils.dataloader import load_dexnet2, ImageDataset

from gqcnn.gqcnn import gqcnn, gqcnn_with_attention
import matplotlib.pyplot as plt

if __name__=='__main__':

    # config 
    opt = options()
    print (opt)

    # cuda
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs("sample_demo/%s" % (opt.name), exist_ok=True)

    # network
    if opt.attention:
        net = gqcnn_with_attention(im_size=32)
    else:
        net = gqcnn(im_size=32)

    print ('Evaluate the project %s' % opt.name)

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

    path_model = "saved_models/%s/model_latest.pth" % (opt.name)
    net.load_state_dict(torch.load(path_model), strict=False)
    print ('load {}'.format(path_model))
    
    # save the result on the text file
    f = open('sample_demo/%s/result.csv' % (opt.name), 'w')

    TN=0
    TP=0
    FN=0
    FP=0

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
            # fig.savefig('sample_demo/%s/%s' % (opt.name, str(i)))
            # release memory
            plt.clf()
            plt.close()

            if opt.attention:
                net.save_attention_mask(images, z, './sample_demo/%s/attn_%d.png' % (opt.name, i))


        correct = (grasp == torch.max(outputs, 1)[1])
 
        data_corrects += torch.sum(correct)

        prob = torch.nn.Softmax(dim=1)(outputs)
        print ('Id: %d, prob: %.5f [%s], T/F: %s'% (i, prob[0,1].item(), gq.item()>opt.gamma, correct.item()))
        f.write('%d, %.8f, %.8f, %s\n'% (i, prob[0,1].item(), gq.item(), correct.item()))

        # 真陽性
        if gq.item()>opt.gamma and correct.item()==True:
            TP += 1
        # 真陰性
        if gq.item()<opt.gamma and correct.item()==True:
            TN += 1
        # 偽陽性
        if gq.item()>opt.gamma and correct.item()==False:
            FP += 1
        if gq.item()<opt.gamma and correct.item()==False:
            FN += 1
        # 偽陰性
        if i >= opt.n_sample-1:
            break

    f.close()

    print ('Correct rate:', ((data_corrects/opt.n_sample).item() * 100), '[%]')
    print ('TP: %d, TN: %d, FP: %d, FN: %d' % (TP, TN, FP, FN))

    accuracy  = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall    =  TP/(TP+FN)
    f_measure = 2*precision*recall/(precision+recall)
    print ('- Accuracy: %.5f' % accuracy )
    print ('- Precision: %.5f' % precision)
    print ('- Recall: %.5f' % recall)
    print ('- F-measure: %.5f' % f_measure)
