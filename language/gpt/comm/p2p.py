import os
import time

import torch
import torch.distributed as dist
import subprocess

rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])
local_world_size = int(world_size/int(os.environ['SLURM_JOB_NUM_NODES']))

node_list = os.environ['SLURM_NODELIST']
addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
os.environ["MASTER_ADDR"] = addr #"127.0.0.2"
os.environ["MASTER_PORT"] = "28524"
os.environ['RANK'] = str(rank)
os.environ['WORLD_SIZE'] = str(world_size)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
# device = torch.device("cuda", rank)


# for i in range(world_size):
#     if i == rank:
#         # print("hahha", rank)
#         print_and_sleep(rank, 1)
#     torch.distributed.barrier()

# tensor = torch.tensor([rank], dtype=torch.int)
# if rank == 0:
#     torch.cuda.synchronize()
#     print(rank)
#     dist.send(tensor, dst=rank+1)
# elif rank == world_size - 1:
#     dist.recv(tensor, src=rank-1)
#     torch.cuda.synchronize()
#     print(rank)
# else:
#     dist.recv(tensor, src=rank-1)
#     torch.cuda.synchronize()
#     print(rank)
#     dist.send(tensor, dst=rank+1)

tensor = torch.tensor([rank], dtype=torch.int)
if rank > 0:
    dist.recv(tensor, src=rank-1)

# torch.cuda.synchronize(rank)
# dist.barrier()
print(rank, local_rank)

if rank < world_size - 1:
    dist.send(tensor, dst=rank+1)

if rank == world_size - 1:
    print("OK")