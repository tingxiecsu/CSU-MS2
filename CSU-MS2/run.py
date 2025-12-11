import os
import argparse
import random
import subprocess
import numpy as np
import yaml
import torch.multiprocessing as mp
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def set_cuda_visible_devices():

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_ids = input('Please select the GPU IDs (e.g., 0,1,2 for 3 GPUs): ')
        
        clean_ids = [g.strip() for g in gpu_ids.split(',') if g.strip()]
        if not clean_ids:
            print("No valid GPU ID provided. Exiting.")
            exit(1)
            
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(clean_ids)
        num_gpus_to_use = len(clean_ids)
        print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}. Will use {num_gpus_to_use} GPU(s).")
        return num_gpus_to_use
    else:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        clean_ids = [g.strip() for g in gpu_ids.split(',') if g.strip()]
        return len(clean_ids)

try:
    WORLD_SIZE_FROM_USER = set_cuda_visible_devices()
except Exception as e:
    print(f"Error during GPU selection: {e}")
    exit(1)

from train import SimCLR
from dataloader.dataset_wrapper import DataSetWrapper
import torch.distributed as dist
import torch

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(deterministic)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def main(rank, world_size, rank_is_set, ds_args):
    
    if not rank_is_set:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12346'
        os.environ['RANK'] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        
    print(f'Process {rank}: Using local GPU index {rank} (Mapped to original ID via CUDA_VISIBLE_DEVICES)')

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    seed_rank = torch.distributed.get_rank()
    
    device = torch.device("cuda", rank)
    seed = 2023 + seed_rank
    set_random_seed(seed, deterministic=False)
    
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = DataSetWrapper(world_size, rank, config['batch_size'], **config['dataset'])

    simclr = ModalTrain(dataset, config, device, world_size, rank)
    simclr.train()

if __name__ == "__main__":
    
    os.environ["NCCL_TIMEOUT"] = "4800"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--moldiff_config', type=str, default='MolDiff/configs/train/train_MolDiff.yml')
    ds_args = parser.parse_args()

    if "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank_is_set = True
    else:
        rank_is_set = False
    world_size = WORLD_SIZE_FROM_USER 
    
    if not rank_is_set:
        if world_size > 1:
            args = (world_size, rank_is_set, ds_args)
            mp.spawn(main, args=args, nprocs=world_size, join=True)
        else:
            rank = 0
            main(rank, world_size, rank_is_set, ds_args)
    else:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print('RANK', rank)
        print('WORLD_SIZE', world_size)
        main(rank, world_size, rank_is_set, ds_args)
