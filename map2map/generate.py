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

from .data.gen_fields import GenerateFieldDataset
from .data import norms
from . import models
from .utils import import_attr, load_model_state_dict

def node_worker(args):
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

    generate_dataset = GenerateFieldDataset(
        style_pattern=args.style_pattern,
        pk_pattern=args.pk_pattern,
        out_pattern=args.out_pattern,
        num_mesh_1d=args.num_mesh_1d,
        device = device,
        sphere_mode=args.sphere_mode,
        crop=args.crop,
    )

    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = generate_dataset.style_size
    in_chan = generate_dataset.in_chan
    out_chan = in_chan

    model = import_attr("nbody.StyledVNet", models)
    model = model(style_size, in_chan, out_chan)
    model.to(device)
    state = torch.load(os.path.dirname(__file__) + "/model_parameters/nbody_params.pt", map_location=device)
    load_model_state_dict(model, state['model'])
    model.eval()

    generate(generate_loader, d2d_model, v2v_model, device, args.no_dis, args.no_vel)

def generate(generate_loader, d2d_model, v2v_model, device, no_dis, no_vel) :

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with torch.no_grad() :

        remainder = generate_loader.dataset.nout % world_size
        num_mocks = generate_loader.dataset.nout // world_size + 1 if rank < remainder else generate_loader.dataset.nout // world_size
        start_mock = rank * num_mocks if rank < remainder else rank * num_mocks + remainder
        end_mock = start_mock + num_mocks
        if rank == 0 :
            pbar = tqdm(total = num_mocks)
            pbar.set_description(f"Generating linear fields")
        for mock in range(start_mock, end_mock) :
            generate_loader.dataset.generate_linear_field(mock, device)
            if rank == 0 :
                pbar.update(1)


        if d2d_model is not None and v2v_model is not None :
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
                jvp_dir = torch.tensor([1.]).to(device)
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
