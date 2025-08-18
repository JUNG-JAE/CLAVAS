import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision


class VGG11Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.encoder == 'vgg11':
            vgg = torchvision.models.vgg11(weights=None)
        elif args.encoder == 'vgg11_bn':
            vgg = torchvision.models.vgg11_bn(weights=None)
        self.features = vgg.features  # CNN layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)          # [B, C, H', W']
        x = self.pool(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)       # [B, 512]
        
        return x


class ProjectionHeadV2(nn.Module):
    # 3-layer MLP with BN & ReLU as in SimCLR v2.
    # Returns (z, h), where h is the penultimate representation used for linear eval in v2.
    # v1 uses only a CNN encoder.
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False) # Since BN is applied, its affine parameters subsume the bias term, we set bias=False
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        h = F.relu(x, inplace=True)  # h: representation before final layer (used for linear eval in v2) 
        z = self.fc3(h)
        
        return z, h


class SimCLRv2Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = VGG11Encoder(args)
        self.proj_head = ProjectionHeadV2(in_dim=512, hidden_dim=args.hidden_dim, out_dim=args.proj_dim)

    def forward(self, x):
        feats = self.encoder(x)       # [B, 512]
        z, h = self.proj_head(feats)  # z: [B, proj_dim], h: [B, hidden_dim]
        # z = F.normalize(z, dim=1) # L2 normalize z for cosine similarity
        
        return z, h


class LinearEvalNet(nn.Module):
    # Freeze encoder + projection head, use h (penultimate) as features for linear classifier.
    def __init__(self, simclr_model: SimCLRv2Model, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        for p in simclr_model.parameters():
            p.requires_grad = False # freeze backbone and projection head
        self.backbone = simclr_model
        self.fc = nn.Linear(hidden_dim, num_classes)

    @torch.no_grad()
    def extract_h(self, x):
        # Return h from projection head; ensure model in eval mode outside
        _z, h = self.backbone(x)
        
        return h  # [B, hidden_dim]

    def forward(self, x):
        with torch.no_grad():
            _z, h = self.backbone(x)
        logits = self.fc(h)
        
        return logits





