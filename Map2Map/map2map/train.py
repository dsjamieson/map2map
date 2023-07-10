import os
import socket
import time
import datetime
import sys
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.func import jvp

from .data import FieldDataset, DistFieldSampler
from . import models
from .models import narrow_cast, resample, lag2eul
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power, plt_xpower

ckpt_link = 'checkpoint.pt'

def node_worker(args):
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node

    node = int(os.environ['SLURM_NODEID'])

    if args.gpus_per_node < 1:
        raise RuntimeError('GPU not found on node {}'.format(node))

    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)

def gpu_worker(local_rank, node, args):

    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    device = torch.device('cuda', 0)

    rank = args.gpus_per_node * node + local_rank

    torch.manual_seed(args.seed + rank)

    dist_file = os.path.join(os.getcwd(), 'dist_addr')
    dist.init_process_group(
        backend=args.dist_backend,
        init_method='file://{}'.format(dist_file),
        world_size=args.world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=14400)
    )
    dist.barrier()
    if rank == 0:
        os.remove(dist_file)

    train_dataset = FieldDataset(
        style_pattern=args.train_style_pattern,
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    train_sampler = DistFieldSampler(train_dataset, shuffle=True,
                                     div_data=args.div_data,
                                     div_shuffle_dist=args.div_shuffle_dist)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=0, #args.loader_workers,
        pin_memory=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            style_pattern=args.val_style_pattern,
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=False,
            aug_shift=None,
            aug_add=None,
            aug_mul=None,
            crop=args.crop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop,
            crop_step=args.crop_step,
            in_pad=args.in_pad,
            tgt_pad=args.tgt_pad,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        val_sampler = DistFieldSampler(val_dataset, shuffle=False,
                                       div_data=args.div_data,
                                       div_shuffle_dist=args.div_shuffle_dist)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=0, #args.loader_workers,
            pin_memory=True,
        )

    args.style_size = train_dataset.style_size
    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan[:1]

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(args.style_size, sum(args.in_chan), sum(args.out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    criterion = import_attr(args.criterion, nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        [
            {
                'params': (param for name, param in model.named_parameters()
                           if 'mlp' in name or 'style' in name),
                'betas': (0.9, 0.99), 'weight_decay': 1e-4,
            },
            {
                'params': (param for name, param in model.named_parameters()
                           if 'mlp' not in name and 'style' not in name),
            },
        ],
        lr=args.lr,
        **args.optimizer_args,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.scheduler_args)

    if (args.load_state == ckpt_link and not os.path.isfile(ckpt_link)
            or not args.load_state):
        if args.init_weight_std is not None:
            model.apply(init_weights)

        for w in model.parameters() :
            dist.broadcast(w, src=0)

        start_epoch = 0

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state['epoch']
        #start_epoch = 0

        load_model_state_dict(model, state['model'],
                              strict=args.load_state_strict)

        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']

            print('state at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state), flush=True)

        del state

    torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        print('pytorch {}'.format(torch.__version__))
        pprint(vars(args))
        sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader, model, criterion,
                           optimizer, scheduler, logger, device, args)

        epoch_loss = train_loss

        if args.val :
            val_loss = validate(epoch, val_loader, model, criterion,
                                logger, device, args)

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_loss[4])

        if rank == 0:
            logger.flush()

            if min_loss is None or epoch_loss[4] < min_loss:
                min_loss = epoch_loss[4]

            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }

            state_file = 'state_{}.pt'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()
    
def train(epoch, loader, model, criterion,
          optimizer, scheduler, logger, device, args):

    model.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)

    if rank == 0 :
        pbar = tqdm(total = len(loader))
        pbar.set_description(f"Training epoch {epoch}")

    for i, data in enumerate(loader) :

        batch = epoch * len(loader) + i + 1

        input = data['input'].to(device, non_blocking=True)
        Om = data['Om'].float().to(device)
        Dz = data['Dz'].float().to(device)

        model_eval = lambda tDz : model(input, Om, tDz)
        jvp_dir = torch.tensor([1.]).to(device)
        (dis_out, s), (vel_out, ds) = jvp(model_eval, (Dz,), (jvp_dir,))

        dis_tgt, vel_tgt = data['target'].to(device, non_blocking=True).split(3, dim = 1)

        if (hasattr(model, 'scale_factor')
                and model.scale_factor != 1):
            input = resample(input, model.scale_factor, narrow=False)
        input, dis_out, vel_out, dis_tgt, vel_tgt= narrow_cast(input, dis_out, vel_out, dis_tgt, vel_tgt)
        
        dis_loss = criterion(dis_out, dis_tgt)
        vel_loss = criterion(vel_out, vel_tgt)
        
        den_out, den_tgt = lag2eul([dis_out, dis_tgt], **args.misc_kwargs)
        mom_out, mom_tgt = lag2eul([dis_out, dis_tgt], val=[vel_out, vel_tgt], **args.misc_kwargs)

        den_loss = criterion(den_out, den_tgt)
        mom_loss = criterion(mom_out, mom_tgt)
        
        loss = (dis_loss * vel_loss) ** 3 * den_loss * mom_loss
        
        epoch_loss[0] += dis_loss.detach()
        epoch_loss[1] += vel_loss.detach()
        epoch_loss[2] += den_loss.detach()
        epoch_loss[3] += mom_loss.detach()
        epoch_loss[4] += loss.detach()

        optimizer.zero_grad()
        torch.log(loss).backward()  # NOTE actual loss is log(loss)
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        dist.barrier()
        optimizer.step()

        if batch % args.log_interval == 0:
            dist.all_reduce(dis_loss)
            dist.all_reduce(vel_loss)
            dist.all_reduce(den_loss)
            dist.all_reduce(mom_loss)
            dist.all_reduce(loss)
            dis_loss /= world_size
            vel_loss /= world_size
            den_loss /= world_size
            mom_loss /= world_size
            loss /= world_size
            if rank == 0:
                logger.add_scalar('loss/batch/train/dis', dis_loss.item(), global_step=batch)
                logger.add_scalar('loss/batch/train/vel', vel_loss.item(), global_step=batch)
                logger.add_scalar('loss/batch/train/den', den_loss.item(), global_step=batch)
                logger.add_scalar('loss/batch/train/mom', mom_loss.item(), global_step=batch)
                logger.add_scalar('loss/batch/train/tot', loss.item(), global_step=batch)
                grads = get_grads(model)
                logger.add_scalar('grad/first', grads[0], global_step=batch)
                logger.add_scalar('grad/last', grads[-1], global_step=batch)
    
        if rank == 0 :
            pbar.update(1)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0 :

        pbar.close()

        logger.add_scalar('loss/epoch/train/dis', epoch_loss[0], global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/vel', epoch_loss[1], global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/den', epoch_loss[2], global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/mom', epoch_loss[3], global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/tot', epoch_loss[4], global_step=epoch+1)

        fig = plt_slices(input[-1], 
                         dis_out[-1], dis_tgt[-1], dis_out[-1] - dis_tgt[-1],
                         vel_out[-1], vel_tgt[-1], vel_out[-1] - vel_tgt[-1],
                         den_out[-1], den_tgt[-1], den_out[-1] - den_tgt[-1],
                         mom_out[-1], mom_tgt[-1], mom_out[-1] - mom_tgt[-1],
            title=['in', 
                   'dis_out', 'dis_tgt', 'dis_err',
                   'vel_out', 'vel_tgt', 'vel_err',
                   'den_out', 'den_tgt', 'den_err',
                   'mom_out', 'mom_tgt', 'mom_err'],
                **args.misc_kwargs,
        )
        logger.add_figure('fig/train', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_xpower(dis_out, dis_tgt, input, Dz)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/train/power/dis', fig, global_step=epoch+1)
        fig.clf()
        
        fig = plt_xpower(vel_out, vel_tgt, input)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/train/power/vel', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_xpower(den_out, den_tgt)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/train/power/den', fig, global_step=epoch+1)
        fig.clf()
        
        fig = plt_xpower(mom_out, mom_tgt)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/train/power/mom', fig, global_step=epoch+1)
        fig.clf()

    return epoch_loss


def validate(epoch, loader, model, criterion, logger, device, args):
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)

    if rank == 0 :
        pbar = tqdm(total=len(loader), file=sys.stdout)
        pbar.set_description(f"Validation epoch {epoch}")

    with torch.no_grad() :
        for batch, data in enumerate(loader) :
        
            input = data['input'].to(device, non_blocking=True)
            Om = data['Om'].float().to(device, non_blocking=True)
            Dz = data['Dz'].float().to(device, non_blocking=True)

            model_eval = lambda tDz : model(input, Om, tDz)
            jvp_dir = torch.tensor([1.]).to(device)
            (dis_out, s), (vel_out, ds) = jvp(model_eval, (Dz,), (jvp_dir,))

            dis_tgt, vel_tgt = data['target'].to(device, non_blocking=True).split(3, dim = 1)

            if (hasattr(model, 'scale_factor')
                    and model.scale_factor != 1):
                input = resample(input, model.scale_factor, narrow=False)
            input, dis_out, vel_out, dis_tgt, vel_tgt= narrow_cast(input, dis_out, vel_out, dis_tgt, vel_tgt)
            
            dis_loss = criterion(dis_out, dis_tgt)
            vel_loss = criterion(vel_out, vel_tgt)
            
            den_out, den_tgt = lag2eul([dis_out, dis_tgt], **args.misc_kwargs)
            mom_out, mom_tgt = lag2eul([dis_out, dis_tgt], val=[vel_out, vel_tgt], **args.misc_kwargs)
            den_loss = criterion(den_out, den_tgt)
            mom_loss = criterion(mom_out, mom_tgt)
            
            loss = (dis_loss * vel_loss) ** 3 * den_loss * mom_loss
            
            epoch_loss[0] += dis_loss.detach()
            epoch_loss[1] += vel_loss.detach()
            epoch_loss[2] += den_loss.detach()
            epoch_loss[3] += mom_loss.detach()
            epoch_loss[4] += loss.detach()

            if rank == 0 :
                pbar.update(1)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:

        pbar.close()

        logger.add_scalar('loss/epoch/val/dis', epoch_loss[0], global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/vel', epoch_loss[1], global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/den', epoch_loss[2], global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/mom', epoch_loss[3], global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/tot', epoch_loss[4], global_step=epoch+1)

        fig = plt_slices(input[-1], 
                         dis_out[-1], dis_tgt[-1], dis_out[-1] - dis_tgt[-1],
                         vel_out[-1], vel_tgt[-1], vel_out[-1] - vel_tgt[-1],
                         den_out[-1], den_tgt[-1], den_out[-1] - den_tgt[-1],
                         mom_out[-1], mom_tgt[-1], mom_out[-1] - mom_tgt[-1],
            title=['in', 
                   'dis_out', 'dis_tgt', 'dis_err',
                   'vel_out', 'vel_tgt', 'vel_err',
                   'den_out', 'den_tgt', 'den_err',
                   'mom_out', 'mom_tgt', 'mom_err'],
                **args.misc_kwargs,
        )
        logger.add_figure('fig/train', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_xpower(dis_out, dis_tgt, input, Dz)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/val/power/dis', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_xpower(vel_out, vel_tgt, input)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/val/power/vel', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_xpower(den_out, den_tgt)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/val/power/den', fig, global_step=epoch+1)
        fig.clf()
        
        fig = plt_xpower(mom_out, mom_tgt)
        fig.subplots_adjust(hspace = 0.03)
        logger.add_figure('fig/val/power/mom', fig, global_step=epoch+1)
        fig.clf()

    return epoch_loss

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(0.0, args.init_weight_std)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.affine:
            # NOTE: dispersion from DCGAN, why?
            m.weight.data.normal_(1.0, args.init_weight_std)
            m.bias.data.fill_(0)

def get_grads(model):
    """gradients of the weights of the first and the last layer
    """
    grads = list(p.grad for n, p in model.named_parameters()
                 if '.weight' in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm() for g in grads]
    return grads
