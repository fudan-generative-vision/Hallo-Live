"""
Occupy about 90% of every visible CUDA GPU and keep it busy.

Usage:
    python tools/occupy_gpu.py
"""

import torch
import os
import torch.multiprocessing as mp
import time


def check_mem(cuda_device):
    devices_info = (
        os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader')
        .read()
        .strip()
        .split("\n")
    )
    total, used = devices_info[int(cuda_device)].split(",")
    return total, used


def loop(cuda_device):
    cuda_i = torch.device(f"cuda:{cuda_device}")
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    occupy_mem = max_mem - used

    bytes_per_mat = 3 * 512 * 512 * 4  # x + y + result = 3
    batch_size = occupy_mem * 1024**2 // bytes_per_mat

    print(
        f"Device number: {cuda_device} | Memory upper limit (90%): {max_mem} MB | Already used: {used} MB | To be occupied: {occupy_mem} MB"
    )

    while True:
        x = torch.rand(batch_size, 512, 512, dtype=torch.float, device=cuda_i)
        y = torch.rand(batch_size, 512, 512, dtype=torch.float, device=cuda_i)
        time.sleep(0.1)  # If the GPU utilization is not high enough, reduce this value.
        x = torch.matmul(x, y)


def main():
    if torch.cuda.is_available():
        num_processes = torch.cuda.device_count()
        processes = list()
        for i in range(num_processes):
            p = mp.Process(target=loop, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
