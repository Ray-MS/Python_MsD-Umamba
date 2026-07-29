from pathlib import Path

import torch
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader
from datetime import datetime

from datasets import create_dataset
from losses import CombinedSegmentationLoss
from models import create_model

__all__ = ['run',]

device = torch.device('cuda')


from tqdm import tqdm
def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.
    for step in tqdm(loader):
        x, y = step[0].to(device), step[1].to(device).squeeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.
    for step in loader:
        x, y = step[0].to(device), step[1].to(device).squeeze(1)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def fit(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs,
    result_dir,
    start_valid_epoch=1,
    valid_interval=1,
) -> None:
    if start_valid_epoch < 1:
        raise ValueError('Start validation epoch must be positive.')
    if valid_interval < 1:
        raise ValueError('Validation interval must be positive.')

    result_dir.mkdir(parents=True, exist_ok=True)
    model = model.cuda()
    loss_fn = CombinedSegmentationLoss().to(device)

    e0 = 0
    best_valid_loss = float('inf')
    best_valid_epoch = 0
    checkpoint_path = result_dir/'train.pth'
    best_checkpoint_path = result_dir/'best.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        e0 = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
        best_valid_epoch = checkpoint.get('best_valid_epoch', 0)

    for epoch in range(e0, epochs):
        ep_idx = epoch + 1
        is_best = False
        print('[{}]epoch=>{}:'.format(datetime.now().isoformat(), ep_idx))
        train_loss = train(model, train_loader, optimizer, loss_fn)
        print('\ttrain=>loss:{}'.format(train_loss))

        should_validate = (
            ep_idx >= start_valid_epoch
            and (ep_idx - start_valid_epoch) % valid_interval == 0
        )
        if should_validate:
            valid_loss = validate(model, valid_loader, loss_fn)
            print('\tvalid=>loss:{}'.format(valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_epoch = ep_idx
                is_best = True
                print('\tbest valid=>epoch:{}, loss:{}'.format(
                    best_valid_epoch, best_valid_loss))

        scheduler.step(epoch)

        data = {
            'epoch': ep_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_valid_loss': best_valid_loss,
            'best_valid_epoch': best_valid_epoch,
        }
        torch.save(data, checkpoint_path)
        if is_best:
            torch.save(data, best_checkpoint_path)
        if ep_idx % 50 == 0:
            torch.save(data, result_dir / f'model_epoch_{ep_idx:03d}.pth')


def run(
    args,
) -> None:
    data_root = args.input
    results_root = args.output
    datasets = args.datasets.split(',')
    models = args.models.split(',')

    for dataset_name in datasets:
        train_set, cfg = create_dataset(
            dataset_name,
            root=data_root,
            train=True,
            keys='training',
        )
        valid_set, _ = create_dataset(
            dataset_name,
            root=data_root,
            train=False,
            keys='validation',
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

        for model_name in models:
            model_name = model_name.lower()
            model = create_model(model_name, **cfg)
            optimizer = create_optimizer(args, model)
            scheduler, _ = create_scheduler(args, optimizer)

            fit(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=args.epochs,
                result_dir=Path(results_root, dataset_name, model_name),
                start_valid_epoch=args.start_valid_epoch,
                valid_interval=args.valid_interval,
            )
