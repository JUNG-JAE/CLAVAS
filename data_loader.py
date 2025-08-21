import random
from PIL import Image
import torchvision
from torchvision import transforms
from dataclasses import dataclass
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset

""" 
# without negatives
class GaussianBlur(object):
    # Gaussian blur augmentation as in SimCLR.
    # Kernel size is small for CIFAR-10 (e.g., 3 or 5).
    def __init__(self, kernel_size: int = 3, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        
        return torchvision.transforms.functional.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigma)
    

class SimCLRTransform:
    # Return two differently augmented views of the same image.
    def __init__(self, image_size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # strength 0.5~1.0
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616)),
        ])

    def __call__(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        
        return xi, xj


class TwoCropCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root=root, train=train, transform=None, download=download)
        self.two_crop_transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        xi, xj = self.two_crop_transform(img)
        return xi, xj


def get_pretrain_loaders(args):
    transform = SimCLRTransform(image_size=32)
    train_set = TwoCropCIFAR10(root=args.data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    
    return loader


def build_linear_eval_transforms():
    # For linear evaluation: standard (weak) augmentation

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),])


def get_supervised_loaders(args):
    transform_train = build_linear_eval_transforms()
    transform_test = build_linear_eval_transforms()

    train_set = CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
    test_set = CIFAR10(root=args.data_dir, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_set, batch_size=args.linear_eval_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.linear_eval_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, test_loader
"""


class GaussianBlur(object):
    # Gaussian blur augmentation as in SimCLR.
    # Kernel size is small for CIFAR-10 (e.g., 3 or 5).
    def __init__(self, kernel_size: int = 3, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return torchvision.transforms.functional.gaussian_blur(
            img, kernel_size=self.kernel_size, sigma=sigma
        )


class SimCLRTransform:
    # Return two differently augmented views of the same image.
    def __init__(self, image_size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # strength 0.5~1.0
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2470, 0.2435, 0.2616)),
        ])

    def __call__(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj


def build_linear_eval_transforms():
    # For linear evaluation or "original(weak)" view: no strong aug, only tensor + norm
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])


class TwoCropCIFAR10(CIFAR10):
    """
    기본: (aug, aug) 두 뷰 반환.
    hard_negative=True 이고 orig_transform이 주어지면,
    확률적으로 (aug, orig) 쌍도 섞어서 반환하여 '원본(비증강)'도 학습에 포함.
    """
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=False,
        hard_negative=False,
        orig_transform=None,
        hard_negative_mix_p=0.5,
    ):
        super().__init__(root=root, train=train, transform=None, download=download)
        self.two_crop_transform = transform
        self.hard_negative = hard_negative
        self.orig_transform = orig_transform
        # hard_negative_mix_p: (aug, aug) 를 반환할 확률.
        # 1 - p 확률로 (aug, orig) 를 반환하여 원본을 포함.
        self.hn_mix_p = float(hard_negative_mix_p)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])

        if self.hard_negative and self.orig_transform is not None:
            # p 확률로 평소처럼 (aug, aug), (1-p) 확률로 (aug, orig)
            if random.random() < self.hn_mix_p:
                xi, xj = self.two_crop_transform(img)
            else:
                xi, _ = self.two_crop_transform(img)  # 하나만 증강
                xj = self.orig_transform(img)         # 원본(약한 변환만)
            return xi, xj

        # 기본 동작: (aug, aug)
        xi, xj = self.two_crop_transform(img)
        return xi, xj


def get_pretrain_loaders(args):
    transform = SimCLRTransform(image_size=32)

    # 안전하게 인자 읽기 (없으면 기본 False/0.5)
    use_hard_neg = getattr(args, 'hard_negative', False)
    mix_p = getattr(args, 'hard_negative_mix_p', 0.5)

    # hard_negative인 경우에만 원본(weak) 변환을 준비
    orig_tf = build_linear_eval_transforms() if use_hard_neg else None

    train_set = TwoCropCIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
        hard_negative=use_hard_neg,
        orig_transform=orig_tf,
        hard_negative_mix_p=mix_p,
    )

    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader


def get_supervised_loaders(args):
    transform_train = build_linear_eval_transforms()
    transform_test = build_linear_eval_transforms()

    train_set = CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
    test_set = CIFAR10(root=args.data_dir, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_set, batch_size=args.linear_eval_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.linear_eval_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader
