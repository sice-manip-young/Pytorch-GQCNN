import os
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gqcnn.gqcnn import gqcnn, weights_init_normal

from utils.options import options
from utils.dataloader import ImageDataset
from utils.graphplot import plot, plot_ch

if __name__=='__main__':

    # cuda
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config 
    opt = options()
    print (opt)
    
    os.makedirs("saved_models/%s" % (opt.dataset_name), exist_ok=True)

    # network
    net = gqcnn(im_size=32)
    if opt.epoch != 0:
        net.load_state_dict(torch.load('saved_models/%s/model_%d.pth' % (opt.dataset_name, opt.epoch)))
    else:
        net.apply(weights_init_normal)
    
    gamma = opt.gamma

    if cuda:
        net.cuda()

    # criterion (loss) and optimizer
    criterion    = torch.nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-04)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # data loader
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5.0)), 
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    }

    image_datasets = {
        'train': ImageDataset(opt, s_data=0, n_data=opt.n_dataset, transforms_=data_transforms['train']), # ID: 0~999
        'val' : ImageDataset(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset, transforms_=data_transforms['val']),  # ID: 1000~1100
    }

    dataloader = DataLoader(
        image_datasets['train'],
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        image_datasets['val'],
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
    )

    # --- train --- # 
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # -- train
        net.train()
        for i, (images, z, gq) in enumerate(dataloader):

            images, z, gq = images.to(device), z.to(device), gq.to(device)

            gq = gq.view(gq.size()[0], -1)
            grasp = torch.where(gq>gamma, 0, 1)
            grasp = grasp.squeeze_()
            
            optimizer.zero_grad()
            outputs = net (images, z)

            loss = criterion (outputs, grasp)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            print("\rEpoch [%3d/%3d] Iter [%5d/%5d], Loss: %f"
                            % (
                    epoch+1,
                    opt.n_epochs,
                    i+1,
                    len(dataloader),
                    train_loss/float(i+1),
                ), end=""
            )

        ave_train_loss = train_loss / len(dataloader)

        # -- evaluate
        net.eval()
        with torch.no_grad():
            for i, (images, z, gq) in enumerate(val_dataloader):
                val_loss += loss.item()
            ave_val_loss = val_loss / len(val_dataloader)    
        print(" - val_loss: %f, time: %.2f"
                % (
                    ave_val_loss,
                    (time.time() - prev_time),
                )
        )

        lr_scheduler.step()
        prev_time = time.time()

        train_loss_list.append(ave_train_loss)
        val_loss_list.append(ave_val_loss)

        torch.save(net.state_dict(), "saved_models/%s/model_latest.pth" % (opt.dataset_name))
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(net.state_dict(), "saved_models/%s/model_%d.pth" % (opt.dataset_name, epoch))

        # plot_ch(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_check=True)

    # plot(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_save=True)