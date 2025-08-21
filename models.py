import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -----------------------------
# Backbones
# -----------------------------
class VGG11Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.encoder in ('vgg11', 'vgg11_bn'):
            vgg = torchvision.models.vgg11_bn(weights=None) if args.encoder.endswith('_bn') else torchvision.models.vgg11(weights=None)
        else:
            # Fallback to BN variant if unspecified
            vgg = torchvision.models.vgg11_bn(weights=None)
        self.features = vgg.features                  # CNN layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))      # Global average pooling to be robust to input size

    def forward(self, x):
        x = self.features(x)          # [B, C, H', W']
        x = self.pool(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)       # [B, 512]
        return x


class ResNetCIFAREncoder(nn.Module):
    """
    ResNet encoder adapted for CIFAR-10:
    - First conv: 3x3, stride=1, padding=1
    - Remove the initial maxpool
    Feature dims (after global pooling):
      - resnet18/34: 512
      - resnet50/101: 2048
    """
    def __init__(self, args):
        super().__init__()
        name = args.encoder.lower()
        if name == 'resnet18':
            backbone = torchvision.models.resnet18(weights=None)
        elif name == 'resnet34':
            backbone = torchvision.models.resnet34(weights=None)
        elif name == 'resnet50':
            backbone = torchvision.models.resnet50(weights=None)
        elif name == 'resnet101':
            backbone = torchvision.models.resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet encoder: {args.encoder}")

        # CIFAR-10 stem tweak
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        # Keep only the convolutional trunk (up to layer4)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Record output dim for convenience
        self.out_dim = 2048 if name in ('resnet50', 'resnet101') else 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [B, out_dim]
        return x


def build_encoder(args) -> nn.Module:
    enc = args.encoder.lower()
    if enc.startswith('vgg11'):
        return VGG11Encoder(args)
    elif enc.startswith('resnet'):
        return ResNetCIFAREncoder(args)
    else:
        raise ValueError(f"Unknown encoder '{args.encoder}'. Use one of "
                         f"['vgg11', 'vgg11_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101'].")


def encoder_out_dim(args) -> int:
    enc = args.encoder.lower()
    if enc.startswith('vgg11'):
        return 512
    if enc in ('resnet18', 'resnet34'):
        return 512
    if enc in ('resnet50', 'resnet101'):
        return 2048
    raise ValueError(f"Unknown encoder '{args.encoder}'.")


# -----------------------------
# Heads & Wrappers
# -----------------------------
class ProjectionHeadV2(nn.Module):
    """3-layer MLP with BN & ReLU as in SimCLR v2.
    Returns (z, h), where h is the penultimate representation used for linear eval in v2.
    """
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
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
        h = F.relu(x, inplace=True)  # penultimate features
        z = self.fc3(h)
        return z, h


class SimCLRv2Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = build_encoder(args)
        in_dim = encoder_out_dim(args)
        self.proj_head = ProjectionHeadV2(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.proj_dim)

    def forward(self, x):
        feats = self.encoder(x)
        z, h = self.proj_head(feats)
        return z, h


class LinearEvalNet(nn.Module):
    """Freeze encoder + projection head, use h (penultimate) as features for linear classifier."""
    def __init__(self, simclr_model: SimCLRv2Model, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        for p in simclr_model.parameters():
            p.requires_grad = False
        self.backbone = simclr_model
        self.fc = nn.Linear(hidden_dim, num_classes)

    @torch.no_grad()
    def extract_h(self, x):
        _z, h = self.backbone(x)
        return h  # [B, hidden_dim]

    def forward(self, x):
        with torch.no_grad():
            _z, h = self.backbone(x)
        logits = self.fc(h)
        return logits
