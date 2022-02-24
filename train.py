import argparse
import os
import time

import torch

from utils.options import options
# from utils.dataloader import dataloader
from gqcnn.gqcnn import gqcnn
from utils.graphplot import plot, plot_ch

if __name__=='__main__':

    # cuda
    cuda = True if torch.cuda.is_available() else False

    # config 
    opt = options()
    print (opt)

    # network
    net = gqcnn(im_size=32)
    print (gqcnn)

    if cuda:
        net.cuda()


    # criterion (loss) and optimizer
    loss         = torch.nn.MSELoss() # square l2 loss
    optimizer    = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-04)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)


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

        # train
        net.train()
        for i, (image, z) in enumerate():

            optimizer.zero_grad()


            print("\rEpoch [%3d/%3d] Iter [%5d/%5d], Loss: %f, Acc: %f, Lr: %f"
                            % (
                    epoch+1,
                    opt.n_epochs,
                    i+1,
                    len(dataloader),
                    train_loss/float(i+1),
                    train_acc/float((i+1)*opt.batch_size),
                    optimizer.param_groups[0]["lr"],
                ), end=""
            )

        ave_train_loss = train_loss / len(dataloader)
        ave_train_acc = train_acc / len(dataloader.dataset)

        net.eval()
        with torch.no_grad():
            for i, (image, z) in enumerate():
                val_loss += loss.item()
            ave_val_loss = val_loss / len(val_dataloader)    
            ave_val_acc = val_acc / len(val_dataloader.dataset)
        print(" - val_loss: %f, val acc: %f, time: %.2f"
                % (
                    ave_val_loss,
                    ave_val_acc,
                    (time.time() - prev_time),
                )
        )

        lr_scheduler.step()
        prev_time = time.time()

        train_loss_list.append(ave_train_loss)
        train_acc_list.append(ave_train_acc)
        val_loss_list.append(ave_val_loss)
        val_acc_list.append(ave_val_acc)

        torch.save(net.state_dict(), "saved_models/%s/%s_model_latest.pth" % (opt.dataset_name, opt.name))
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(net.state_dict(), "saved_models/%s/%s_model_%d.pth" % (opt.dataset_name, opt.name, epoch))

        # 途中経過確認用
        plot_ch(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_check=True)

    plot(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_save=True)