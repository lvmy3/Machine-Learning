import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

def Load_cifar10(dir, input_size, batch_size, val_ratio=0.1):
    data_transforms = {
        'train': T.Compose([
            T.RandomCrop(input_size, padding=4, pad_if_needed=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'val': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
    data_train = dset.CIFAR10(dir, train=True, download=True, transform=data_transforms['train'])
    data_val = dset.CIFAR10(dir, train=True, download=True, transform=data_transforms['val'])
    data_test = dset.CIFAR10(dir, train=False, download=True, transform=data_transforms['test'])
    
    num_train = len(data_train)
    split = int(np.floor(num_train * val_ratio))
    loader_train = DataLoader(data_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split, num_train)))
    loader_val = DataLoader(data_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split)))
    loader_test = DataLoader(data_test, batch_size=batch_size)
    dataloaders = {
        'train': loader_train,
        'val': loader_val,
        'test': loader_test
    }
    return dataloaders

def Load_cifar100(dir, input_size, batch_size, val_ratio=0.1):
    data_transforms = {
        'train': T.Compose([
            T.RandomCrop(input_size, padding=4, pad_if_needed=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
        ]),
        'val': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
        ])
    }
    data_train = dset.CIFAR100(dir, train=True, download=True, transform=data_transforms['train'])
    data_val = dset.CIFAR100(dir, train=True, download=True, transform=data_transforms['val'])
    data_test = dset.CIFAR100(dir, train=False, download=True, transform=data_transforms['test'])
    
    num_train = len(data_train)
    split = int(np.floor(num_train * val_ratio))
    loader_train = DataLoader(data_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split, num_train)))
    loader_val = DataLoader(data_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split)))
    loader_test = DataLoader(data_test, batch_size=batch_size)
    dataloaders = {
        'train': loader_train,
        'val': loader_val,
        'test': loader_test
    }
    return dataloaders
 
def Load_imagenet(dir, input_size, batch_size, val_ratio=0.1):
    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    data_train = dset.ImageFolder(dir, transform=data_transforms['train'])
    data_val = dset.ImageFolder(dir, transform=data_transforms['val'])
    data_test = dset.ImageFolder(dir, transform=data_transforms['test'])
    
    num_train = len(data_train)
    split = int(np.floor(num_train * val_ratio))
    loader_train = DataLoader(data_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split, num_train)))
    loader_val = DataLoader(data_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split)))
    loader_test = DataLoader(data_test, batch_size=batch_size)
    dataloaders = {
        'train': loader_train,
        'val': loader_val,
        'test': loader_test
    }
    return dataloaders

def Load_mnist(dir, input_size, batch_size, val_ratio=0.1):
    data_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
    data_train = dset.MNIST(dir, train=True, download=True, transform=data_transforms)
    data_val = dset.ImageFolder(dir, train=True, download=True, transform=data_transforms)
    data_test = dset.ImageFolder(dir, train=True, download=True, transform=data_transforms)
    
    num_train = len(data_train)
    split = int(np.floor(num_train * val_ratio))
    loader_train = DataLoader(data_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split, num_train)))
    loader_val = DataLoader(data_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(split)))
    loader_test = DataLoader(data_test, batch_size=batch_size)
    dataloaders = {
        'train': loader_train,
        'val': loader_val,
        'test': loader_test
    }
    return dataloaders
