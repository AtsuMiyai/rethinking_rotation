import os
from torchvision import datasets, transforms
from datasets.ImageFolder_tin import ImageFolder
from datasets.ImageFolder_cifar import CIFAR100
import pickle

num_CIFAR100 = 50000
num_TIN = 100000


def get_transform(image_size=None):
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=False):
    train_transform, test_transform = get_transform(image_size=image_size)
    if dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        if P.mode == 'simclr'  or P.mode == 'sup_linear':
            nega_posi_flag = [-1 for i in range(num_CIFAR100)]
        elif P.mode == 'simclr_pda':
            nega_posi_flag = [1 for i in range(num_CIFAR100)]
        elif P.mode == 'simclr_nda':
            nega_posi_flag = [0 for i in range(num_CIFAR100)]
        elif P.mode == 'simclr_pnda':
            assert P.nega_posi_flag!='', "please add --nega_posi_flag"
            f = open(P.nega_posi_flag, "rb")
            nega_posi_flag = pickle.load(f)
            nega_posi_flag = list(nega_posi_flag.values())
        train_set = CIFAR100(P.data, train=True, nega_posi_flag=nega_posi_flag, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(P.data, train=False, download=download, transform=test_transform)

    elif dataset == 'tiny_imagenet':
        image_size = (64, 64, 3)
        n_classes = 200
        if P.mode == 'simclr' or P.mode == 'sup_linear':
            nega_posi_flag = [-1 for i in range(num_TIN)]
        elif P.mode == 'simclr_pda':
            nega_posi_flag = [1 for i in range(num_TIN)]
        elif P.mode == 'simclr_nda':
            nega_posi_flag = [0 for i in range(num_TIN)]
        elif P.mode == 'simclr_pnda':
            assert P.nega_posi_flag!='', "please add --nega_posi_flag"
            f = open(P.nega_posi_flag, "rb")
            nega_posi_flag = pickle.load(f)
            nega_posi_flag = list(nega_posi_flag.values())

        train_dir = os.path.join(P.data, 'train')
        test_dir = os.path.join(P.data, 'val')
        train_set = ImageFolder(train_dir, nega_posi_flag, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        raise NotImplementedError()
    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes
