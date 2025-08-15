import logging

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.aug_utils import RandomCropPaste, MixUp, CutMix
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import RandAugment


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # 基础变换
    base_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 训练集变换
    transform_train_auto = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        RandAugment(num_ops=2, magnitude=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 测试集变换
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        if args.random_aug:
            trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train_auto)
        else:
            trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    # 创建数据增强实例
    random_crop_paste = RandomCropPaste(size=args.img_size,alpha=args.cut_rate,flip_p=args.flip_p)
    mixup = MixUp(alpha=args.mixup_rate)
    cutmix = CutMix(size=args.img_size, beta=args.cutmix_rate)

    class AugmentedDataLoader:
        def __init__(self, loader):
            self.loader = loader
            self.aug_method = args.aug_type

        def __iter__(self):
            for batch in self.loader:
                x, y = batch
                # 随机选择一种数据增强方法
                if self.aug_method == 'random_crop_paste':
                    x = random_crop_paste(x)
                    yield x, y
                elif self.aug_method == 'mixup':
                    x, y_a, y_b, lam = mixup((x, y))
                    lam = torch.tensor(lam, device=x.device)
                    yield x, y_a, y_b, lam
                elif self.aug_method == 'cutmix':
                    x, y, y_rand, lam = cutmix((x, y))
                    lam = torch.tensor(lam, device=x.device)
                    yield x, y, y_rand, lam
                elif self.aug_method == 'batch_random':
                    aug_method = np.random.choice(['random_crop_paste', 'mixup', 'cutmix'])
                    print(aug_method)
                    if aug_method == 'random_crop_paste':
                        x = random_crop_paste(x)
                        yield x, y
                    elif aug_method == 'mixup':
                        x, y_a, y_b, lam = mixup((x, y))
                        lam = torch.tensor(lam, device=x.device)
                        yield x, y_a, y_b, lam
                    elif aug_method == 'cutmix':
                        x, y, y_rand, lam = cutmix((x, y))
                        lam = torch.tensor(lam, device=x.device)
                        yield x, y, y_rand, lam
                else:
                    yield x, y
                

        def __len__(self):
            return len(self.loader)

    # 使用增强后的数据加载器
    train_loader = AugmentedDataLoader(train_loader)

    return train_loader, test_loader
