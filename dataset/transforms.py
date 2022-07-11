from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import random
from PIL import ImageFilter, ImageOps

CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
COLORJITTER_DEFAULT = 0.3

# CIFAR DEFAULT TRAIN/TEST TRANSFORMS
DEFAULT_CIFAR_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
    ])
DEFAULT_CIFAR_TEST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
    ])

# DeiT Aug
class GaussianBlur(object):
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

DEIT3_TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            gray_scale(p=1),
            Solarization(p=1.0),
            GaussianBlur(p=1.0)
        ]),
        transforms.ColorJitter(COLORJITTER_DEFAULT, COLORJITTER_DEFAULT, COLORJITTER_DEFAULT),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
    ])