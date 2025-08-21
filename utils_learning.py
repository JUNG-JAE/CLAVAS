import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

import os 
import numpy as np
import random

from data_loader import get_pretrain_loaders, get_supervised_loaders
from models import SimCLRv2Model, LinearEvalNet
from utils_system import print_log

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def nt_xent_loss(z_i, z_j, tau=0.5):
    # Normalized Temperature-scaled Cross Entropy Loss. / Returns: scalar loss
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t())      # [2B, 2B]
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device) # mask self-similarity
    sim = sim / tau
    sim.masked_fill_(mask, -9e15)

    targets = torch.arange(2 * batch_size, device=z.device) # positives: for i in [0..B-1], pos index = i+B; for i in [B..2B-1], pos index = i-B
    targets = (targets + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, targets)
    
    return loss


def pretrain(args, logger):
    set_seed(args.seed)
    device = torch.device(args.device)
    
    loader = get_pretrain_loaders(args)
    model = SimCLRv2Model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # if os.path.exists(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}.pth'):
    #     print_log(logger, f"Loading checkpoint from {args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs} for pretraining...")
    #     payload = torch.load(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}.pth', map_location=args.device)
    #     state = payload.get("state_dict", payload)
    #     model.load_state_dict(state, strict=True)
    # else:
    #     print_log(logger, "Checkpoint not found; using in-memory model.")

    losses = []
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for (x_i, x_j) in loader:
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            z_i, _ = model(x_i)
            z_j, _ = model(x_j)

            loss = nt_xent_loss(z_i, z_j, tau=args.tau)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        
        losses.append(avg_loss)
        
        print_log(logger, f"[Pretrain] Epoch {epoch:03d}/{args.epochs}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save encoder + projection head
    os.makedirs(os.path.dirname(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}.pth'), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": args.__dict__,}, f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}.pth')
    print_log(logger, f"Saved checkpoint to: {f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}.pth'}")
    
    return model, losses


def accuracy(logits, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item() * 100.0)
    
    return res


def linear_eval(args, model: SimCLRv2Model, logger):
    device = torch.device(args.device)
    train_loader, test_loader = get_supervised_loaders(args)

    model.eval()

    lin = LinearEvalNet(model, hidden_dim=args.hidden_dim, num_classes=10).to(device) # num_class 나중에 수정할 것
    optimizer = optim.SGD(lin.parameters(), lr=args.linear_eval_lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.linear_eval_epochs)
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(1, args.linear_eval_epochs + 1):
        lin.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = lin(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc1, = accuracy(logits, y, topk=(1,))
            n = x.size(0)
            train_loss += loss.item() * n
            train_acc += acc1 * n
            total += n

        scheduler.step()
        train_loss /= total
        train_acc /= total

        # Eval
        lin.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = lin(x)
                pred = logits.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
        test_acc = 100.0 * test_correct / test_total
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print_log(logger, f"[LinearEval] Epoch {epoch:03d}/{args.linear_eval_epochs}  TrainLoss: {train_loss:.4f}  TrainAcc@1: {train_acc:.2f}%  TestAcc@1: {test_acc:.2f}%")

    return lin, train_acc_list, test_acc_list
