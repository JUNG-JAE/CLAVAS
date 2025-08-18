import random
from PIL import Image
import torchvision
from torchvision import transforms
from dataclasses import dataclass
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset


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