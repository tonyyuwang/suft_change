import os
import torch
import torch.distributed



"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = True
dist_rank = 0
world_size = 1


def set_gpu_mode(mode, local_rank):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    # local_rank = torch.distributed.get_rank()
    print("local_rank", local_rank)
    # local_rank = int(os.environ["LOCAL_RANK"])
    # dist_rank = 0
    # print("os.environ:", os.environ.get('CUDA_VISIBLE_DEVICES'))
    world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
    print("world_size:", world_size)
    distributed = world_size > 1  # on
    use_gpu = mode

    print("gpu_id", gpu_id)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
