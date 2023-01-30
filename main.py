# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections import OrderedDict

from torch import nn
import torch
import torchvision

from inat import iNatDataset
from sop import SOPDataset
from cub200 import Cub200Dataset
from lars import LARS
from transform import Transform, RandAugment, EasyTransform


DATASETS = [
    'imagenet',
    'sop',
    'inat',
    'cub200',
]

BACKBONES = [
    'resnet18',
    'resnet50',
]

TRANSFORMS = [
    'easy',
    'base',
    'randaugment',
]

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='sop', choices=DATASETS, help='Training dataset')
parser.add_argument('--backbone', type=str, default='resnet50', choices=BACKBONES, help='Neural network')
parser.add_argument('--pretrained', action='store_true', default=False, help='Run in distributed')
parser.add_argument('--freeze_bn', action='store_true', default=False, help='Run in distributed')
parser.add_argument('--transform', type=str, default='base', choices=TRANSFORMS, help='Set of transform')
parser.add_argument('--workers', default=10, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        n_gpu_per_node = world_size // n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)

        # define whether this is the master process / if we are in distributed mode
        is_master = node_id == 0 and local_rank == 0
        multi_node = n_nodes > 1
        multi_gpu = world_size > 1

        if is_master:
            PREFIX = "%i - " % global_rank
            print(PREFIX + "Number of nodes: %i" % n_nodes)
            print(PREFIX + "Node ID        : %i" % node_id)
            print(PREFIX + "Local rank     : %i" % local_rank)
            print(PREFIX + "Global rank    : %i" % global_rank)
            print(PREFIX + "World size     : %i" % world_size)
            print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
            print(PREFIX + "Master         : %s" % str(is_master))
            print(PREFIX + "Multi-node     : %s" % str(multi_node))
            print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
            print(PREFIX + "Hostname       : %s" % socket.gethostname())

        args.rank = int(os.getenv('SLURM_NODEID')) * n_gpu_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.dist_url = 'env://'

        main_worker(local_rank, args)

    else:
        # single-node distributed training
        # import socket
        # sock = socket.socket()
        # sock.bind(('', 0))
        # sock.getsockname()[1]
        args.rank = 0
        args.dist_url = 'tcp://localhost:58471'
        args.world_size = args.ngpus_per_node
        torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir = Path(os.path.expandvars(os.path.expanduser(args.checkpoint_dir)))
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_backbone_weights = []
    param_backbone_biases = []
    param_projector_weights = []
    param_projector_biases = []
    for param in model.backbone.parameters():
        if param.ndim == 1:
            param_backbone_biases.append(param)
        else:
            param_backbone_weights.append(param)
    for param in model.projector.parameters():
        if param.ndim == 1:
            param_projector_biases.append(param)
        else:
            param_projector_weights.append(param)
    parameters = [{'params': param_backbone_weights}, {'params': param_backbone_biases}, {'params': param_projector_weights}, {'params': param_projector_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    scaler = torch.cuda.amp.GradScaler()

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
    else:
        start_epoch = 0

    if args.transform == 'base':
        transform_fn = Transform()
    elif args.transform == 'easy':
        transform_fn = EasyTransform()
    elif args.transform == 'randaugment':
        transform_fn = RandAugment()

    if args.dataset == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(args.data / 'train', transform_fn)
    elif args.dataset == 'sop':
        dataset = SOPDataset(args.data, transform=transform_fn)
    elif args.dataset == 'inat':
        dataset = iNatDataset(args.data, transform=transform_fn)
    elif args.dataset == 'cub200':
        dataset = Cub200Dataset(args.data, transform=transform_fn)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), *_) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), args=args)
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
            if (epoch+1) % 1 == 0:
                state = dict(epoch=epoch + 1, model=model.state_dict(), args=args,
                             optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), backbone=model.module.backbone.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}.pth')
    if args.rank == 0:
        # save final model
        state = dict(epoch=epoch + 1, model=model.state_dict(), args=args,
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), backbone=model.module.backbone.state_dict())
        torch.save(state, args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights #* int(step >= warmup_steps)
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases #* int(step >= warmup_steps)
    optimizer.param_groups[2]['lr'] = lr * args.learning_rate_weights #* 5
    optimizer.param_groups[3]['lr'] = lr * args.learning_rate_biases #* 5


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adapt_checkpoint(state_dict: OrderedDict, remove: str = 'module.', replace: str = '') -> OrderedDict:
    """
    This function renames keys in a state_dict.
    The default function is helpfull when a NN has been used with parallelism.
    """
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        new_key = key.replace(remove, replace)
        new_dict[new_key] = weight
    return new_dict


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet18':
            self.backbone = torchvision.models.resnet18(zero_init_residual=True, pretrained=args.pretrained)
        elif args.backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(zero_init_residual=True, pretrained=args.pretrained)
        num_features = self.backbone.fc.weight.size(1)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [num_features] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        if args.pretrained and (args.backbone == 'resnet50'):
            barlow_state = torch.load('barlow_state.pth', map_location='cpu')
            self.load_state_dict(adapt_checkpoint(barlow_state['model']))

    def forward(self, y1, y2):
        if self.args.freeze_bn:
            self.backbone.eval()

        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


if __name__ == '__main__':
    main()
