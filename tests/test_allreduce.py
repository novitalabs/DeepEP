"""
Minimal 2-node NCCL all-reduce test.
Usage (run on each node respectively):
  Node 0: MASTER_ADDR=10.137.0.223 MASTER_PORT=8361 WORLD_SIZE=2 RANK=0 python tests/test_allreduce.py
  Node 1: MASTER_ADDR=10.137.0.223 MASTER_PORT=8361 WORLD_SIZE=2 RANK=1 python tests/test_allreduce.py
"""
import os
import time
import torch
import torch.distributed as dist


def worker(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    torch.cuda.set_device(local_rank)

    print(f'[rank {global_rank}] Initializing process group (node={node_rank}, local_rank={local_rank}, '
          f'world_size={world_size}, master={ip}:{port}) ...', flush=True)

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=global_rank,
    )

    print(f'[rank {global_rank}] Process group initialized.', flush=True)

    # ---------- simple all-reduce test ----------
    tensor = torch.ones(1024, device=f'cuda:{local_rank}') * global_rank
    expected_sum = sum(range(world_size))  # 0+1+2+...+(N-1)

    dist.barrier()
    t0 = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000

    ok = torch.allclose(tensor, torch.full_like(tensor, expected_sum))
    print(f'[rank {global_rank}] all_reduce result: {"PASS" if ok else "FAIL"}  '
          f'(expected={expected_sum}, got={tensor[0].item():.0f}, time={elapsed_ms:.2f} ms)',
          flush=True)

    # ---------- bandwidth test (optional, 256 MB) ----------
    big = torch.randn(64 * 1024 * 1024, device=f'cuda:{local_rank}')  # 256 MB
    dist.barrier()

    n_iters = 10
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    total_s = time.time() - t0

    bw = big.numel() * big.element_size() * 2 * (world_size - 1) / world_size * n_iters / total_s / 1e9
    if local_rank == 0:
        print(f'[node {node_rank}] all_reduce bandwidth: {bw:.2f} GB/s  '
              f'({n_iters} iters, {total_s * 1000:.1f} ms total)', flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f'[rank {global_rank}] Done.', flush=True)


if __name__ == '__main__':
    num_local_ranks = int(os.getenv('LOCAL_WORLD_SIZE', 8))
    torch.multiprocessing.spawn(worker, args=(num_local_ranks,), nprocs=num_local_ranks)
