import os
from torchvision import datasets, transforms


def get_transform(image_size=None):
    if image_size:
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):
    train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(P.data, train=True, download=download, transform=train_transform)
        infer_set = datasets.CIFAR100(P.data, train=True, download=download, transform=test_transform)

    elif dataset == 'tiny_imagenet':
        image_size = (64, 64, 3)
        n_classes = 200
        train_dir = os.path.join(P.data, 'train')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        infer_set = datasets.ImageFolder(train_dir, transform=test_transform)
    else:
        raise NotImplementedError()

    return train_set, infer_set, image_size, n_classes
