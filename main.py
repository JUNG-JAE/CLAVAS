# The script includes:
# - SimCLR data augmentations (two views), RotNet data augmentations
# - VGG11 encoder adapted for 32x32 with AdaptiveAvgPool
# - 3-layer projection head (SimCLR v2, RotNet)
# - NT-Xent loss implementation
# - Self-supervised pretraining loop for SimCLR and RotNet
# - Linear evaluation (freeze encoder + projection head, train a linear classifier on h)
# - t-SNE visualization utilities

import argparse

from utils_learning import simCLR_pretrain, rotnet_pretrain, linear_eval
from utils_system import set_logger, print_log

from data_loader import get_supervised_loaders
from utils_system import tsne_from_args_and_loader, plot_linear_eval_acc, plot_pretrain_losses

def main():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='select the experiment dataset')
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--num_workers', type=int, default=4, help='number of subprocesses to use for data loading')
    
    # Model
    parser.add_argument('--encoder', type=str, default='vgg11_bn', help='set the model')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or not')
    parser.add_argument('--seed', type=int, default=42, help='seed for model parameters')
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--proj_dim', type=int, default=128)
    
    # Learning for pretraining
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    # parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature parameter')
    
    # Learning for linear evaluation
    parser.add_argument('--linear_eval_batch', type=int, default=512, help='batch size for linear evaluation')
    parser.add_argument('--linear_eval_epochs', type=int, default=100, help='number of epochs for linear evaluation')
    parser.add_argument('--linear_eval_lr', type=float, default=1e-2, help='learning rate for linear evaluation')
    
    # System
    parser.add_argument('--type', type=str, default='SSL', help='CL: Contrastive Learning, SSL: Self-Supervised Learning')
    parser.add_argument('--log_dir', type=str, default='./runs', help='log file path')
    args = parser.parse_args()
    
    logger = set_logger(args)
    
    print_log(logger, "Arguments")
    kv = vars(args)
    width = max(len(k) for k in kv) if kv else 0
    for k in iter(kv):
        print_log(logger, f"  {k:<{width}} : {kv[k]}")
    
    if args.type == 'CL': # Contrastive Learning
        trained_encoder, pretrain_losses = simCLR_pretrain(args, logger)
    elif args.type == 'SSL': # Self-Supervised Learning
        trained_encoder, pretrain_losses = rotnet_pretrain(args, logger)

    _, train_acc_list, test_acc_list = linear_eval(args, trained_encoder, logger)
    
    plot_pretrain_losses(args, pretrain_losses)
    plot_linear_eval_acc(args, train_acc_list, test_acc_list)
    
    _, test_loader = get_supervised_loaders(args)  # CIFAR-10 data loaders with labels
    tsne_from_args_and_loader(args, trained_encoder, test_loader, mode="encoder", max_samples=5000, perplexity=30.0, n_iter=1000) # 2D t-SNE for encoder output
    tsne_from_args_and_loader(args, trained_encoder, test_loader, mode="h", max_samples=5000, perplexity=30.0, n_iter=1000) # 2D t-SNE for h output
    
    return 0


if __name__ == "__main__":
    main()