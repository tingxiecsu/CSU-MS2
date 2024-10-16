from train import SimCLR
import yaml
from dataloader.dataset_wrapper import DataSetWrapper
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse
import random
import subprocess
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def set_random_seed(seed, deterministic=False):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(deterministic)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def get_gpu_count():
    try:
        # nvidia-smi
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
        # Parse the output to get the GPU count
        gpu_count = len(result.stdout.decode('utf-8').strip().split('\n'))
        return gpu_count
    except Exception as e:
        print(f"Error getting GPU count: {e}")
        return 0

def main(rank, world_size, num_gpus, rank_is_set, ds_args):
    import torch
    if not rank_is_set:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12346'
        os.environ['RANK'] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
    print(f'Confirm GPU:{num_gpus}-{rank}')

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        seed_rank = torch.distributed.get_rank()
    else:
        seed_rank = 0
    device = torch.device("cuda", rank)
    seed = 2023 + seed_rank
    set_random_seed(seed, deterministic=False)
    
    config = yaml.load(open(".../config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = DataSetWrapper(world_size,rank, config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config,device,world_size,rank)
    simclr.train()


if __name__ == "__main__":
    world_size = get_gpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="num of gpus")
    parser.add_argument('--moldiff_config', type=str, default='MolDiff/configs/train/train_MolDiff.yml')
    # parser = deepspeed.add_config_arguments(parser)
    ds_args = parser.parse_args() 
    if "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank_is_set = True
    else:
        rank_is_set = False
    if not rank_is_set:
        if ds_args.num_gpus == -1:
            num_gpus = int(input('Number of Gpus used: '))
        else:
            num_gpus = ds_args.num_gpus
        if num_gpus == world_size:
            print('Let\'s use', world_size, 'GPUs!')
        elif num_gpus < world_size:
            gpu_ids = input('Please select the GPU number: ')
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            import torch
            assert len(gpu_ids.split(',')) == num_gpus
            world_size = torch.cuda.device_count()
            assert world_size == num_gpus
        else:
            raise
        if world_size > 1:
            args = (world_size, num_gpus, rank_is_set, ds_args)
            mp.spawn(main, args=args, nprocs=world_size, join=True)
        else:
            rank = 0
            main(rank, world_size, num_gpus, rank_is_set, ds_args)
    else:
        rank = int(os.environ["LOCAL_RANK"])
        print('RANK',rank)
        world_size = int(os.environ["WORLD_SIZE"])
        print('WORLD_SIZE',world_size)
        num_gpus = ds_args.num_gpus
        main(rank, world_size, num_gpus, rank_is_set, ds_args)
 
