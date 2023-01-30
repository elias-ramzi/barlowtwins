from pathlib import Path
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from inat import iNatDataset
from sop import SOPDataset
from cub200 import Cub200Dataset
from get_knn import get_knn


def filter_checkpoint(state_dict: OrderedDict, filter_on: str) -> OrderedDict:
    """
    this function filters a state_dict using the `filter_on` key.

    ex:
    state_dict = OrderedDict([backbone.weight, backbone.bias, head.weight, head.bias])
    filter_checkpoint(state_dict, 'backbone.') -> OrderedDict([weight, bias])
    """
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        if key.startswith(filter_on):
            new_dict[key.replace(filter_on, '')] = weight
    return new_dict


def create_label_matrix(labels, other_labels=None, dtype=torch.int64):
    labels = labels.squeeze()
    other_labels = (other_labels if other_labels is not None else labels).squeeze()

    return (labels.unsqueeze(1) == other_labels).squeeze().to(dtype)


def recall_rate_at_k(sorted_target, at_k, reduction='mean'):
    r_at_k = sorted_target[:, :at_k].any(1).float()

    if reduction == 'mean':
        return r_at_k.mean()
    else:
        return r_at_k


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--checkpoint', required=True, type=Path, help='path to checkpoint directory')
parser.add_argument('--data', type=Path, default=None, help='possible override of data path')
parser.add_argument('--workers', default=10, type=int, help='number of data loader workers')
parser.add_argument('--batch-size', default=2048, type=int, help='mini-batch size')
parser.add_argument('--metric-bs', default=256, type=int, help='mini-batch size for metrics (per worker)')
parser.add_argument('--dist', action='store_true', default=False, help='Run in distributed')
parser.add_argument('--recalls', type=int, default=[1, 10, 100], nargs='+', help='recalls')


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count() if args.dist else 1
    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    if args.dist:
        print(f"Running in distributed seting world_size={args.ngpus_per_node}")
        torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    else:
        main_worker(0, args)


def main_worker(gpu, args):
    if args.dist:
        args.rank += gpu
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(gpu)

    state = torch.load(args.checkpoint, map_location='cpu')
    state['args'] = argparse.Namespace(
        backbone='resnet50',
        dataset='cub200',
        data='/local/DEEPLEARNING/image_retrieval/CUB_200_2011',

    )
    # state['args'] = argparse.Namespace(
    #     backbone='resnet50',
    #     dataset='sop',
    #     data='/local/DEEPLEARNING/image_retrieval/Stanford_Online_Products',
    #
    # )

    if state['args'].backbone == 'resnet18':
        net = torchvision.models.resnet18(zero_init_residual=True)
    elif state['args'].backbone == 'resnet50':
        net = torchvision.models.resnet50(zero_init_residual=True, pretrained=True)

    net.fc = torch.nn.Identity()
    # net.load_state_dict(torch.load('/share/DEEPLEARNING/datasets/image_retrieval/experiments/barlowtwins/inat_40h/resnet50.pth', map_location='cpu'))
    # net.load_state_dict(state['backbone'])
    net.load_state_dict(filter_checkpoint(state['model'], 'module.backbone.'))
    net.requires_grad_(False)
    net.eval()
    net.to('cuda', non_blocking=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    if state['args'].dataset == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(state['args'].data / 'train', transform)
    elif state['args'].dataset == 'sop':
        dataset = SOPDataset(state['args'].data, 'test', transform=transform)
    elif state['args'].dataset == 'inat':
        dataset = iNatDataset(state['args'].data, 'test', transform=transform)
    elif state['args'].dataset == 'cub200':
        dataset = Cub200Dataset(state['args'].data, 'test', transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if args.dist else None
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler, persistent_workers=False)

    local_features = []
    local_index = []
    for (img, _, index) in tqdm(loader, disable=args.rank != 0):
        with torch.cuda.amp.autocast():
            X = net(img.to('cuda', non_blocking=True))
        X = torch.nn.functional.normalize(X)
        local_features.append(X)
        local_index.append(index)

    local_features = torch.cat(local_features)
    local_index = torch.cat(local_index).to('cuda', non_blocking=True)

    if args.dist:
        tensor_list_features = [torch.empty_like(local_features) for _ in range(dist.get_world_size())]
        tensor_list_index = [torch.empty_like(local_index) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list_features, local_features)
        dist.all_gather(tensor_list_index, local_index)
        features = torch.empty(len(loader.dataset), local_features.size(-1), device=local_features.device, dtype=local_features.dtype)
        all_features = torch.cat(tensor_list_features)
        all_index = torch.cat(tensor_list_index)
        features[all_index] = all_features
        labels = torch.from_numpy(loader.dataset.labels[:, 0]).to('cuda', non_blocking=True)
        local_labels = labels[local_index]
    else:
        features = local_features
        labels = local_labels = torch.from_numpy(loader.dataset.labels[:, 0]).to('cuda', non_blocking=True)

    BS = args.metric_bs
    numk = max(args.recalls) + 1
    num_iter = (local_features.size(0) // BS) + (local_features.size(0) % BS != 0)
    metrics = defaultdict(list)
    for i in tqdm(range(num_iter), disable=args.rank != 0):
        indices = get_knn(
            local_features[i*BS:(i+1)*BS],
            features,
            numk,
            embeddings_come_from_same_source=True,
        )
        sorted_target = create_label_matrix(local_labels[i*BS:(i+1)*BS], labels[indices], dtype=torch.float)
        for R in args.recalls:
            metrics[f'R@{R}'].append(recall_rate_at_k(sorted_target, R, 'none'))

    metrics = {k: torch.cat(v) for k, v in metrics.items()}
    if args.dist:
        local_metrics = torch.stack(list(metrics.values()), 1)
        metrics_list = [torch.empty_like(local_metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_list, local_metrics)
        all_metrics = torch.empty(len(loader.dataset), len(metrics), device=local_metrics.device, dtype=local_metrics.dtype)
        all_metrics[all_index] = torch.cat(metrics_list)
        metrics = {k: v.item() for k, v in zip(metrics.keys(), all_metrics.mean(0))}
    else:
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    if args.rank == 0:
        for k, v in metrics.items():
            print(f"{k} --> {np.around(v*100, 2)}")


if __name__ == '__main__':
    main()
