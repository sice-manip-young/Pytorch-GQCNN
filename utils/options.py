import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--n_dataset", type=int, default=5000, help="number of dataset of training")
    parser.add_argument("--n_valid_dataset", type=int, default=500, help="number of dataset of training")
    parser.add_argument("--dataset_name", type=str, default="dex-net_2.0", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=32, help="size of image height")
    parser.add_argument("--img_width", type=int, default=32, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")

    parser.add_argument("--gamma", type=float, default=0.002, help="gamma threshold")

    parser.add_argument("--n_iterations", type=int, default=3000, help="gamma threshold")
    opt = parser.parse_args()

    return opt