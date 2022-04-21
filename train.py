import os
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gqcnn.gqcnn import gqcnn, gqcnn_with_attention, weights_init_normal

from utils.options import options
from utils.dataloader import ImageDataset, load_dexnet2
from utils.graphplot import plot, plot_ch

if __name__=='__main__':

    # cuda
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config 
    opt = options()
    print (opt)
    
    os.makedirs("saved_models/%s" % (opt.name), exist_ok=True)

    # network
    if opt.attention:
        net = gqcnn_with_attention(im_size=32)
    else:
        net = gqcnn(im_size=32)
    print ('Summary: ', net)
    
    if opt.dual_gpu:
        net = torch.nn.DataParallel(net, device_ids=[0, 1]) # multi-gpu

    if opt.epoch != 0:
        net.load_state_dict(torch.load('saved_models/%s/model_%d.pth' % (opt.name, opt.epoch)))
    else:
        net.apply(weights_init_normal)
    
    gamma = opt.gamma

    if cuda:
        net.cuda()

    # criterion (loss) and optimizer
    criterion    = torch.nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum) # removed weight_decay=5e-04
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

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

    # load from dexnet-2.0
    images, depth ,gquality = load_dexnet2(opt, s_data=0, n_data=opt.n_dataset)
    images_valid, depth_valid , gquality_valid = load_dexnet2(opt, s_data=opt.n_dataset, n_data=opt.n_valid_dataset)

    image_datasets = {
         'train': ImageDataset(images, depth, gquality, transforms_=data_transforms['train']), # ID: 0~999
         'val' : ImageDataset(images_valid, depth_valid, gquality_valid, transforms_=data_transforms['val']),  # ID: 1000~1100
    }

    dataloader = DataLoader(
        image_datasets['train'],
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True, 
        num_workers=opt.n_cpu,
        drop_last=True
    )

    val_dataloader = DataLoader(
        image_datasets['val'],
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )

    # --- train --- # 
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        train_loss = 0
        val_loss = 0
        epoch_corrects = 0

        # -- train
        net.train()
        for i, (images, z, gq) in enumerate(dataloader):
            images, z, gq = images.to(device), z.to(device), gq.to(device)

            optimizer.zero_grad()

            gq = gq.view(gq.size()[0], -1)
            grasp = torch.where(gq>gamma, 1, 0)
            grasp = torch.squeeze(grasp, dim=1)
            
            outputs = net (images, z)

            loss = criterion (outputs, grasp)
            epoch_corrects += torch.sum(grasp == torch.max(outputs, 1)[1])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            print("\rEpoch [%3d/%3d] Iter [%5d/%5d], Loss: %f, Acc: %f"
                            % (
                    epoch+1,
                    opt.n_epochs,
                    i+1,
                    opt.n_iterations, 
                    train_loss/float(i+1),
                    float(epoch_corrects)/(float(i+1)*opt.batch_size)
                ), end=""
            )

            if i >= opt.n_iterations-1:
                break

        ave_train_loss = train_loss / opt.n_iterations * opt.batch_size
        ave_train_acc  = float(epoch_corrects)/float(opt.n_iterations*opt.batch_size)

        # -- evaluate
        net.eval()
        with torch.no_grad():
            epoch_corrects = 0
            for i, (images, z, gq) in enumerate(val_dataloader):
                images, z, gq = images.to(device), z.to(device), gq.to(device)
                gq = gq.view(gq.size()[0], -1)
                grasp = torch.where(gq>gamma, 1, 0)
                grasp = torch.squeeze(grasp, dim=1)

                outputs = net (images, z)
                loss = criterion (outputs, grasp)

                val_loss += loss.item()
                epoch_corrects += torch.sum(grasp == torch.max(outputs, 1)[1])

                if i > opt.n_iterations//10:
                    break
            ave_val_loss = val_loss / (opt.n_iterations//10)
            ave_val_acc  = epoch_corrects / (opt.n_iterations//10)

        print(" - val_loss: %f, time: %.2f"
                % (
                    ave_val_loss,
                    (time.time() - prev_time),
                )
        )

        #if opt.attention:
        #    net.save_attention_mask(images, z, './saved_models/%s/attn_%d.png' % (opt.name, epoch))

        lr_scheduler.step()
        prev_time = time.time()

        train_loss_list.append(ave_train_loss)
        val_loss_list.append(ave_val_loss)
        train_acc_list.append(ave_train_acc)
        val_acc_list.append(ave_val_acc)

        if opt.dual_gpu:
            torch.save(net.module.state_dict(), "saved_models/%s/model_latest.pth" % (opt.name))
        else:
            torch.save(net.state_dict(), "saved_models/%s/model_latest.pth" % (opt.name))

        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            if opt.dual_gpu:
                torch.save(net.module.state_dict(), "saved_models/%s/model_%d.pth" % (opt.name, epoch))
            else:
                torch.save(net.state_dict(), "saved_models/%s/model_%d.pth" % (opt.name, epoch))

        plot_ch(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_check=True)
    plot(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_save=True)
