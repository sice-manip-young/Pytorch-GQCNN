import matplotlib.pyplot as plt

def plot(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_save=True):
    fig=plt.figure()
    plt.plot(range(opt.epoch, opt.n_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(opt.epoch, opt.n_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    
    if is_save:
        fig.savefig('plot_loss.png')


    fig=plt.figure()
    plt.plot(range(opt.epoch, opt.n_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(opt.epoch, opt.n_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()

    if is_save:
        fig.savefig('plot_acc.png')

def plot_ch(opt, train_loss_list, train_acc_list, val_loss_list, val_acc_list, is_check=False):

    n_epoch = len(train_acc_list)

    fig=plt.figure()
    plt.plot(range(opt.epoch, n_epoch), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(opt.epoch, n_epoch), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    
    if is_check:
        fig.savefig('prog_plot_loss.png')

    plt.close()

    fig=plt.figure()
    plt.plot(range(opt.epoch, n_epoch), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(opt.epoch, n_epoch), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()

    if is_check:
        fig.savefig('prog_plot_acc.png')

    plt.close()