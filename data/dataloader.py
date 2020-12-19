import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import SVHN
from .usps import *


def get_data(args):
    if args.task == 's2m':
        src_data = SVHN('../data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        
        tgt_data = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.task == 'u2m':
        src_data = USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        
        tgt_data = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    else:
        src_data = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        tgt_data = USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    
    src_data, tgt_data = relabel_data(src_data, tgt_data, args.task)    

    src_loader = torch.utils.data.DataLoader(src_data, 
                            batch_size=args.batch_size, 
                            shuffle=True, num_workers=0)

    tgt_loader = torch.utils.data.DataLoader(tgt_data,
                            batch_size=args.batch_size, 
                            shuffle=True, num_workers=0)
    return src_loader, tgt_loader

def relabel_data(src_data, tgt_data, task, known_cnum=5):
    image_path = []
    image_label = []
    if task == 's2m':
        for i in range(len(src_data.data)):
            if int(src_data.labels[i]) < known_cnum:
                image_path.append(src_data.data[i])
                image_label.append(src_data.labels[i])
        src_data.data = image_path
        src_data.labels = image_label
    else:
        for i in range(len(src_data.train_data)):
            if int(src_data.train_labels[i]) < known_cnum:
                image_path.append(src_data.train_data[i])
                image_label.append(src_data.train_labels[i])
        src_data.train_data = image_path
        src_data.train_labels = image_label

    for i in range(len(tgt_data.train_data)):
        if int(tgt_data.train_labels[i]) >= known_cnum:
            tgt_data.train_labels[i] = known_cnum

    return src_data, tgt_data