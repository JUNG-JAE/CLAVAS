# Write a complete PyTorch script for SimCLR v2 (VGG11 backbone, CIFAR-10).
# The script includes:
# - SimCLR data augmentations (two views)
# - VGG11 encoder adapted for 32x32 with AdaptiveAvgPool
# - 3-layer projection head (SimCLR v2)
# - NT-Xent loss implementation
# - Self-supervised pretraining loop
# - Linear evaluation (freeze encoder + projection head, train a linear classifier on h)
# - CLI args
# - Basic logging and checkpoint saving
#
# Save to /mnt/data so the user can download.


import os
import argparse

from utils_learning import pretrain, linear_eval
from utils_system import set_logger, print_log


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='select the experiment dataset')
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--num_workers', type=int, default=4, help='number of subprocesses to use for data loading')
    
    # Model
    parser.add_argument('--encoder', type=str, default='vgg11', help='set the model')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or not')
    parser.add_argument('--seed', type=int, default=42, help='seed for model parameters')
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--proj_dim', type=int, default=128)
    
    # Learning for pretraining
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature parameter')
    
    # Learning for linear evaluation
    parser.add_argument('--linear_eval_batch', type=int, default=512, help='batch size for linear evaluation')
    parser.add_argument('--linear_eval_epochs', type=int, default=1, help='number of epochs for linear evaluation')
    parser.add_argument('--linear_eval_lr', type=float, default=1e-2, help='learning rate for linear evaluation')
    
    # System
    parser.add_argument('--type', type=str, default='CL', help='log file path')
    parser.add_argument('--log_dir', type=str, default='./runs', help='log file path')
    # parser.add_argument('--ckpt_path', type=str, default='./runs/vgg11_cifar10.pth')
    args = parser.parse_args()
    
    logger = set_logger(args)
    
    print_log(logger, "Arguments")
    kv = vars(args)
    width = max(len(k) for k in kv) if kv else 0
    for k in iter(kv):
        print_log(logger, f"  {k:<{width}} : {kv[k]}")
    
    trained_encoder = pretrain(args, logger)
    
    linear_eval(args, trained_encoder, logger)
    
    return 0


if __name__ == "__main__":
    main()