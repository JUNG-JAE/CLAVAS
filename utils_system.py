import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, Tuple, Literal
from matplotlib.ticker import MaxNLocator, MultipleLocator


def create_directory(path):
    os.makedirs(path, exist_ok=True)
        

def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def set_logger(args):
    create_directory(f'{args.log_dir}/{args.type}')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    if args.type == 'CL':
        filename=f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}.log'
    elif args.type == 'downstream':
        filename=f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.downstream_epochs}.log'
    
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# ===== t-SNE utilities for visualization =====
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@torch.no_grad()
def _forward_for_embedding(model: nn.Module, x: torch.Tensor, mode: Literal["encoder","h"]="encoder") -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    if hasattr(model, "encoder"):
        # SimCLR v2 스타일
        feats = model.encoder(x)
        if mode == "encoder":
            emb = feats
        elif hasattr(model, "proj_head") and mode == "h":
            _, h = model.proj_head(feats)
            emb = h
        else:
            raise ValueError("mode 'h can only be used with SimCLR v2")
    else:
        if mode != "encoder":
            raise ValueError("Standard feature extractors(e.g., nn.Sequential) support only mode='encoder'")
        out = model(x)              # [B,D] or [B,D,1,1]
        if out.dim() == 4:          # [B,D,1,1]
            out = torch.flatten(out, 1)
        emb = out

    if was_training:
        model.train()

    return emb.detach()


def extract_embeddings_for_tsne(model: nn.Module, loader, device: Optional[str] = None, mode: Literal["encoder","h"]="encoder", max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device.type

    model.eval()
    feats_list, labels_list = [], []
    total = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Loader must yield (images, labels).")

        emb = _forward_for_embedding(model, x, mode=mode)
        feats_list.append(emb.cpu().numpy())
        labels_list.append(y.numpy())

        total += x.size(0)
        if max_samples is not None and total >= max_samples:
            break

    feats_np = np.concatenate(feats_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    if max_samples is not None and feats_np.shape[0] > max_samples:
        feats_np = feats_np[:max_samples]
        labels_np = labels_np[:max_samples]
    return feats_np, labels_np


def run_tsne(feats_np: np.ndarray, labels_np: np.ndarray, save_path: str,
             title: str = "t-SNE (perplexity=30)", perplexity: float = 30.0,
             learning_rate: float = 200.0, n_iter: int = 1000, random_state: int = 42) -> str:
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init="pca",
        metric="euclidean",
        random_state=random_state,
        verbose=1,
    )
    emb2d = tsne.fit_transform(feats_np)

    plt.figure(figsize=(8, 6), dpi=150)
    cmap = plt.get_cmap("tab10")
    classes = np.unique(labels_np)
    for cls in classes:
        idx = labels_np == cls
        plt.scatter(
            emb2d[idx, 0], emb2d[idx, 1], s=8, alpha=0.8,
            label=(CIFAR10_CLASSES[int(cls)] if 0 <= int(cls) < len(CIFAR10_CLASSES) else str(int(cls))),
            c=np.array([cmap(int(cls) % 10)])
        )
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def tsne_from_model_and_loader(model: nn.Module, loader, save_path: str, mode: Literal["encoder","h"]="encoder", max_samples: Optional[int] = 5000,
                               perplexity: float = 30.0, learning_rate: float = 200.0, n_iter: int = 1000, random_state: int = 42, title: Optional[str] = None) -> str:
    feats, labels = extract_embeddings_for_tsne(model=model, loader=loader, mode=mode, max_samples=max_samples)
    
    if title is None:
        title = f"t-SNE ({mode})"
    return run_tsne(
        feats_np=feats, labels_np=labels, save_path=save_path, title=title,
        perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)


def tsne_from_args_and_loader(args, model: nn.Module, loader,
                              mode: Literal["encoder","h"]="encoder", **tsne_kwargs) -> str:
    
    if args.type == 'CL':
        save_path = f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}_tsne_{mode}.png'
    elif args.type == 'downstream':
        save_path = f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.downstream_epochs}_tsne_{mode}.png'
    
    return tsne_from_model_and_loader(model, loader, save_path=save_path, mode=mode, **tsne_kwargs)


""" 
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] # CIFAR10 class names for legends

@torch.no_grad()
def _forward_for_embedding(simclr_model, x: torch.Tensor, mode: Literal["encoder","h"]="encoder") -> torch.Tensor:
    # Return an embedding from a SimCLRv2 model.
    
    was_training = simclr_model.training # Handle both training/eval outside
    simclr_model.eval()
    device = next(simclr_model.parameters()).device

    x = x.to(device, non_blocking=True)

    # Forward
    feats = simclr_model.encoder(x)
    if mode == "encoder":
        emb = feats
    elif mode == "h":
        _, h = simclr_model.proj_head(feats) # Use projection head to get (z, h); return h
        emb = h
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'encoder' or 'h'.")

    if was_training:
        simclr_model.train()

    return emb.detach()


def extract_embeddings_for_tsne(simclr_model, loader, device: Optional[str] = None, mode: Literal["encoder","h"]="encoder", max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(simclr_model.parameters()).device.type

    simclr_model.eval()

    feats_list, labels_list = [], []
    total = 0

    for batch in loader:
        # Expect (x, y)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Loader must yield (images, labels).")

        with torch.no_grad():
            emb = _forward_for_embedding(simclr_model, x, mode=mode)

        feats_list.append(emb.cpu().numpy())
        labels_list.append(y.numpy())

        total += x.size(0)
        if max_samples is not None and total >= max_samples:
            break

    feats_np = np.concatenate(feats_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)

    if max_samples is not None and feats_np.shape[0] > max_samples:
        feats_np = feats_np[:max_samples]
        labels_np = labels_np[:max_samples]

    return feats_np, labels_np


def run_tsne(feats_np: np.ndarray,
             labels_np: np.ndarray,
             save_path: str,
             title: str = "t-SNE (perplexity=30)",
             perplexity: float = 30.0,
             learning_rate: float = 200.0,
             n_iter: int = 1000,
             random_state: int = 42) -> str:

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init="pca",
        metric="euclidean",
        random_state=random_state,
        verbose=1,
    )
    emb2d = tsne.fit_transform(feats_np)

    # Plot
    plt.figure(figsize=(8, 6), dpi=150)
    # Use a stable color map (tab10) for up to 10 classes as in CIFAR-10
    cmap = plt.get_cmap("tab10")

    classes = np.unique(labels_np)
    for cls in classes:
        idx = labels_np == cls
        # Matplotlib expects a color for the whole collection; build a 2D array for consistency
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], s=8, alpha=0.8, label=(CIFAR10_CLASSES[int(cls)] if 0 <= int(cls) < len(CIFAR10_CLASSES) else str(int(cls))), c=np.array([cmap(int(cls) % 10)]))

    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    return save_path


def tsne_from_model_and_loader(simclr_model, loader, save_path: str, mode: Literal["encoder","h"]="encoder", max_samples: Optional[int] = 5000, 
                               perplexity: float = 30.0, learning_rate: float = 200.0, n_iter: int = 1000, random_state: int = 42, title: Optional[str] = None) -> str:
    feats, labels = extract_embeddings_for_tsne(
        simclr_model=simclr_model,
        loader=loader,
        mode=mode,
        max_samples=max_samples
    )
    if title is None:
        title = f"t-SNE ({mode})"

    return run_tsne(
        feats_np=feats,
        labels_np=labels,
        save_path=save_path,
        title=title,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
    )


def tsne_from_args_and_loader(args, simclr_model, loader, mode: Literal["encoder","h"]="encoder", **tsne_kwargs) -> str:
    save_path = f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}_tsne_{mode}.png'
    
    return tsne_from_model_and_loader(simclr_model, loader, save_path=save_path, mode=mode, **tsne_kwargs)
"""

def plot_pretrain_losses(args, losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss per Epoch')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if args.type == 'CL':
        plt.savefig(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}_pretrain_losses.png', dpi=150)
    elif args.type == 'downstream':
        plt.savefig(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.downstream_epochs}_pretrain_losses.png', dpi=150)

def plot_linear_eval_acc(args, train_acc, test_acc):
    if args.type == 'CL':
        assert args.linear_eval_epochs == len(train_acc) == len(test_acc), "length mismatch"
        epoch_list = list(range(1, args.linear_eval_epochs+1))
    elif args.type == 'downstream':    
        assert args.downstream_epochs == len(train_acc) == len(test_acc), "length mismatch"
        epoch_list = list(range(1, args.downstream_epochs+1))
    
    train_y, test_y = train_acc, test_acc
    
    plt.figure()
    # plt.plot(epoch_list, train_y, marker='o', label='Train Acc')
    plt.plot(epoch_list, test_y,  marker='s', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per epoch')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    ax = plt.gca()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    
    plt.tight_layout()
    if args.type == 'CL':
        plt.savefig(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.epochs}_linear_eval_acc.png', dpi=150)
    elif args.type == 'downstream':
        plt.savefig(f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}_{args.downstream_epochs}_linear_eval_acc.png', dpi=150)
        
# ===== Intra-class distance utilities =====
""" 
@torch.no_grad()
def _extract_embeddings(simclr_model, loader, mode: Literal["encoder","h"]="encoder"):
    simclr_model.eval()
    device = next(simclr_model.parameters()).device
    feats_all, labels_all = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        feats = simclr_model.encoder(x)
        if mode == "encoder":
            emb = feats
        elif mode == "h":
            _, h = simclr_model.proj_head(feats)
            emb = h
        else:
            raise ValueError(...)
        feats_all.append(emb.detach().cpu())
        labels_all.append(y.detach().cpu())
    return torch.cat(feats_all, 0), torch.cat(labels_all, 0)
"""

@torch.no_grad()
def _extract_embeddings(model, loader, mode: Literal["encoder","h"]="encoder"):
    # only used for encoder
    model.eval()
    device = next(model.parameters()).device
    feats_all, labels_all = [], []

    has_encoder = hasattr(model, "encoder")
    has_proj_head = hasattr(model, "proj_head")

    if (mode == "h") and not (has_encoder and has_proj_head):
        raise ValueError("The mode='h' option is only supported for SimCLR-style models that have a projection head. For downstream classifiers/Sequential models, please use mode='encoder'")

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)

        if has_encoder:
            feats = model.encoder(x)
            if mode == "encoder":
                emb = feats
            else:  # mode == 'h'
                _, h = model.proj_head(feats)
                emb = h
        else:
            if mode != "encoder":
                raise ValueError("For downstream classifiers/Sequential models, please use mode='encoder'")
            out = model(x)           # [B,D] 또는 [B,D,1,1]
            emb = torch.flatten(out, 1) if out.dim() == 4 else out

        feats_all.append(emb.detach().cpu())
        labels_all.append(y.detach().cpu())

    return torch.cat(feats_all, 0), torch.cat(labels_all, 0)


def compute_intra_class_distances(simclr_model, loader, mode: Literal["encoder","h"]="encoder"):
    feats, labels = _extract_embeddings(simclr_model, loader, mode=mode)
    results = {}
    for cls in torch.unique(labels).tolist():
        X = feats[labels == int(cls)]
        if X.shape[0] == 0:
            results[int(cls)] = float("nan")
            continue
        c = X.mean(dim=0, keepdim=True)
        dists = torch.norm(X - c, dim=1)
        results[int(cls)] = dists.mean().item()
        
    return results


def log_intra_class_distances(logger, distances_dict, prefix: str = "", use_cifar10_names: bool = True):
    parts = []
    for cls_idx in sorted(distances_dict.keys()):
        name = CIFAR10_CLASSES[cls_idx] if (use_cifar10_names and 0 <= cls_idx < len(CIFAR10_CLASSES)) else str(cls_idx)
        parts.append(f"{name}: {distances_dict[cls_idx]:.4f}")

    print_log(logger, parts)


def intra_class_distances(args, simclr_model, loader, mode: Literal["encoder","h"]="encoder", logger=None):
    if logger is None:
        logger = set_logger(args)
    distances = compute_intra_class_distances(simclr_model, loader, mode=mode)
    print_log(logger, f"[IntraClassDistance/{mode}] mean L2 distance to class centroid")
    log_intra_class_distances(logger, distances, prefix=f"[{mode}]")


# ===== For downstream tasks models =====
class _Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def build_resnet_encoder_from_classifier(clf: nn.Module) -> nn.Module:
    assert hasattr(clf, "model"), "classifier.model (torchvision resnet) is required."
    backbone = clf.model  # torchvision resnet
    # children() = [conv1, bn1, relu, maxpool(or Identity), layer1..4, avgpool, fc]
    # fc 제거하고 avgpool까지 살리고, flatten 추가
    modules = list(backbone.children())[:-1]  # drop fc
    encoder = nn.Sequential(*modules, _Flatten())
    
    return encoder
