import os
import torch
import torch.distributed as dist
import subprocess


rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])
node_list = os.environ['SLURM_NODELIST']
addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

if not dist.is_initialized():
    dist.init_process_group(rank=rank, 
                            world_size=world_size, 
                            backend="nccl",
                            init_method=f'tcp://[{addr}]:23332')

# # 在进程0中创建一个张量
# tensor_send = torch.tensor([1, 2, 3, 4]).cuda()

# if rank == 0:
#     # 发送张量到进程1
#     print(rank, " sd")
#     dist.send(tensor_send, dst=1)
# elif rank == 1:
#     # 接收进程0发送的张量
#     print(rank," re")
#     tensor_recv = torch.empty(tensor_send.size()).cuda()
#     dist.recv(tensor_recv, src=0)
#     print("Received tensor:", tensor_recv, rank)

device = torch.device("cpu", rank)

a1 = torch.ones(1, requires_grad=True).to(device)
if rank == 0:
    a1 = torch.tensor([100.], requires_grad=True).to(device)
    dst = 1
    dist.batch_isend_irecv([dist.P2POp(dist.send, a1, dst)])
else:
    a1 = torch.zeros(1, requires_grad=True).to(device)
    dst = 0
    dist.batch_isend_irecv([dist.P2POp(dist.recv, a1, dst)])

torch.cuda.synchronize()

if rank < 2:
    print(f'Rank {rank}: ', a1.cpu())

# torch.cuda.synchronize(rank)
# print(rank, "OK")
# import sys
# sys.exit()