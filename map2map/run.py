import os
import sys
import datetime
import warnings
from pprint import pprint
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.func import jvp

from .data.run_fields import RunFieldDataset
from .data import norms
from . import models
from .utils import import_attr, load_model_state_dict

def node_worker(args):

    if args.no_dis and args.no_vel :
        print("Asked for no_dis and no_vel, nothing to do, exiting")
        exit()

    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    args.threads_per_node = int(os.environ['SLURM_CPUS_ON_NODE'])
    if args.num_threads != -1 :
        if args.threads_per_node > args.num_threads :
            args.threads_per_node = arg.num_threads

    if torch.cuda.is_available():
        args.gpus_per_node = torch.cuda.device_count()
        args.workers_per_node = args.gpus_per_node
        args.world_size = args.nodes * args.gpus_per_node
        args.dist_backend = 'nccl'
    else :
        args.workers_per_node = 1
        args.world_size = args.nodes
        args.dist_backend = 'gloo'
    node = int(os.environ['SLURM_NODEID'])

    spawn(worker, args=(node, args), nprocs=args.workers_per_node)

def worker(local_rank, node, args):

    if torch.cuda.is_available():
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        device = torch.device('cuda', local_rank)
        torch.backends.cudnn.benchmark = True
        rank = args.gpus_per_node * node + local_rank
    else:  # CPU multithreading
        device = torch.device('cpu')
        rank = local_rank
        torch.set_num_threads(args.threads_per_node)

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

        if args.verbose :
            print('pytorch {}'.format(torch.__version__))
            print()
            pprint(vars(args))
            print()
            sys.stdout.flush()

    run_dataset = RunFieldDataset(
        style_pattern=args.style_pattern,
        in_pattern=args.in_pattern,
        out_pattern=args.out_pattern,
        crop=args.crop,
    )

    run_loader = DataLoader(
        run_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = run_dataset.style_size
    in_chan = run_dataset.in_chan
    out_chan = in_chan

    model = import_attr("nbody.NbodyD2DStyledVNet", models)
    model = model(style_size, in_chan, out_chan)
    model.to(device)
    state = torch.load(os.path.dirname(__file__) + "/model_parameters/nbody_params.pt", map_location=device)
    load_model_state_dict(model, state['model'])
    model.eval()

    run(run_loader, model, device, args.no_dis, args.no_vel)

def run(run_loader, model, device, no_dis, no_vel) :

    rank = dist.get_rank()

    with torch.no_grad() :

        if not no_dis and not no_vel :
            write_chan = (3,) * 2
        else :
            write_chan = (3,)

        dis_norm = torch.ones(1, dtype=torch.float64)
        norms.cosmology.dis(dis_norm)
        dis_norm = dis_norm.to(torch.float32).to(device, non_blocking=True)

        if rank == 0 :
            pbar = tqdm(total = len(run_loader))
            pbar.set_description(f"Mapping to nonlinear fields")

        for i, data in enumerate(run_loader):

            input = data['input'].to(device, non_blocking=True)
            input = input * dis_norm
            Om = data['Om'].to(torch.float32).to(device, non_blocking=True)
            Dz = data['Dz'].to(torch.float32).to(device, non_blocking=True)

            if not no_vel :
                model_eval = lambda tDz : model(input, Om, tDz)
                jvp_dir = torch.tensor([1.]).to(device, non_blocking=True)
                (dis_out, s), (vel_out, ds) = jvp(model_eval, (Dz,), (jvp_dir,))

                z, Om = data['redshift'], data['Om']
                vel_norm = torch.ones(1, dtype=torch.float64)
                norms.cosmology.vel(vel_norm, undo=True, Om=Om, z=z)
                vel_norm = vel_norm.to(torch.float32).to(device, non_blocking=True)
                vel_out = vel_out * vel_norm

            else :
                dis_out, s = model(input, Om, Dz)
            
            if not no_dis :
                dis_out = dis_out / dis_norm
            
            out_dir = data['out_dir']
            if not no_dis and not no_vel :
                output = torch.cat((dis_out, vel_out), 1)
                out_paths = [[os.path.join(out_dir[0], 'dis')], [os.path.join(out_dir[0], 'vel')]]
            elif not no_dis : 
                output = dis_out
                out_paths = [[os.path.join(out_dir[0], 'dis')]]
            elif not no_vel :
                output = vel_out
                out_paths = [[os.path.join(out_dir[0], 'vel')]]
            run_loader.dataset.assemble('_out', write_chan, output, out_paths)

            if rank == 0 :
                pbar.update(1)
