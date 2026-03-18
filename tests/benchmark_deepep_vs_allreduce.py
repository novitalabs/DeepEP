"""
Benchmark: DeepEP low-latency dispatch (per-token FP8) vs torch.distributed.all_reduce
Configuration matches sglang production defaults:
  - return_recv_hook=True, async_finish=False (send/recv split for overlap)
  - use_fp8=True, per_token=True
  - EP8, hidden=5120, 160 experts, topk=8

Usage:
    python benchmark_deepep_vs_allreduce.py [--num-tokens 128] [--num-warmups 50] [--num-iters 100]
"""
import argparse
import numpy as np
import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist, per_token_cast_back


def bench_fn(fn, num_warmups=50, num_iters=100):
    """Benchmark a function, return (avg, min, max, median) in seconds."""
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    times = np.array(times[1:])  # drop first
    return np.mean(times), np.min(times), np.max(times), np.median(times)


def run_benchmark(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_tokens = args.num_tokens
    hidden = 5120
    num_experts = 160
    num_topk = 8
    num_local_experts = num_experts // num_ranks

    if rank == 0:
        print(f"{'='*70}")
        print(f"Benchmark: DeepEP LL vs torch.all_reduce (sglang defaults)")
        print(f"  num_ranks={num_ranks}, num_tokens={num_tokens}, hidden={hidden}")
        print(f"  num_experts={num_experts}, num_topk={num_topk}")
        print(f"  num_local_experts={num_local_experts}")
        print(f"  mode: per-token FP8, return_recv_hook=True (sglang prod)")
        print(f"  warmups={args.num_warmups}, iters={args.num_iters}")
        print(f"{'='*70}")
        print(flush=True)

    # --- Setup data ---
    torch.manual_seed(42 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # ===== Setup DeepEP Buffer (matches sglang) =====
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts)
    if rank == 0:
        print(f"DeepEP buffer size: {num_rdma_bytes / 1e6:.1f} MB", flush=True)

    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        explicitly_destroy=True)

    # --- Warm up dispatch to get handle for combine ---
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    use_fp8=True, per_token=True,
                                    async_finish=False, return_recv_hook=True)
    hook()  # complete recv

    packed_recv_x_cont = (packed_recv_x[0], packed_recv_x[1].contiguous())
    simulated_gemm_x = per_token_cast_back(
        packed_recv_x_cont[0].view(-1, hidden),
        packed_recv_x_cont[1].view(-1, 1)
    ).view(packed_recv_x_cont[0].shape)

    # Simulated large GEMM (matches sglang overlap pattern)
    gemm_lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
    gemm_rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')

    # =================================================================
    # Benchmark 1: DeepEP dispatch only (sglang default: hook mode)
    # =================================================================
    # Mode A: return_recv_hook=True (sglang production default)
    #   - kernel 1: send phase only
    #   - hook(): kernel 2: recv phase
    #   - between send and hook, sglang overlaps a large GEMM

    def deepep_dispatch_hook():
        """sglang default: send → hook (no overlap, measures pure comm)"""
        _, _, _, _, recv_hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=True, per_token=True,
            async_finish=False, return_recv_hook=True)
        recv_hook()

    avg_dh, min_dh, max_dh, med_dh = bench_fn(deepep_dispatch_hook, args.num_warmups, args.num_iters)

    # Mode B: return_recv_hook=True + GEMM overlap between send and recv
    def deepep_dispatch_hook_overlap():
        """sglang default with GEMM overlap between send and recv"""
        _, _, _, _, recv_hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=True, per_token=True,
            async_finish=False, return_recv_hook=True)
        # Overlap: large GEMM while waiting for RDMA
        gemm_lhs @ gemm_rhs
        recv_hook()

    avg_dho, min_dho, max_dho, med_dho = bench_fn(deepep_dispatch_hook_overlap, args.num_warmups, args.num_iters)

    # Mode C: single kernel (no hook split, for comparison)
    def deepep_dispatch_single():
        """Single kernel: send+recv in one launch"""
        _, _, _, event, _ = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=True, per_token=True,
            async_finish=True, return_recv_hook=False)
        event.current_stream_wait()

    avg_ds, min_ds, max_ds, med_ds = bench_fn(deepep_dispatch_single, args.num_warmups, args.num_iters)

    # =================================================================
    # Benchmark 2: DeepEP dispatch + combine (full round trip)
    # =================================================================
    # sglang default: both dispatch and combine use return_recv_hook=True
    def deepep_full_hook():
        """Full dispatch+combine with hook mode (sglang default)"""
        recv_x, recv_count, h, _, dispatch_hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=True, per_token=True,
            async_finish=False, return_recv_hook=True)
        dispatch_hook()
        combined_x, _, combine_hook = buffer.low_latency_combine(
            simulated_gemm_x, topk_idx, topk_weights, handle,
            async_finish=False, return_recv_hook=True)
        combine_hook()

    avg_fh, min_fh, max_fh, med_fh = bench_fn(deepep_full_hook, args.num_warmups, args.num_iters)

    # Full round trip with GEMM overlap on both dispatch and combine recv
    def deepep_full_hook_overlap():
        """Full round trip with GEMM overlap (sglang production pattern)"""
        recv_x, recv_count, h, _, dispatch_hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=True, per_token=True,
            async_finish=False, return_recv_hook=True)
        gemm_lhs @ gemm_rhs  # overlap dispatch recv
        dispatch_hook()
        combined_x, _, combine_hook = buffer.low_latency_combine(
            simulated_gemm_x, topk_idx, topk_weights, handle,
            async_finish=False, return_recv_hook=True)
        gemm_lhs @ gemm_rhs  # overlap combine recv
        combine_hook()

    avg_fho, min_fho, max_fho, med_fho = bench_fn(deepep_full_hook_overlap, args.num_warmups, args.num_iters)

    # Compute bandwidth
    num_fp8_bytes_per_token = hidden + 1 * 4 + 16  # FP8 data + 1 scale + alignment
    total_dispatch_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        total_dispatch_bytes += num_fp8_bytes_per_token * num_selections

    if rank == 0:
        print(f"\n--- DeepEP Low-Latency (per-token FP8) ---")
        print(f"  Dispatch (return_recv_hook=True, sglang default):")
        print(f"    no overlap:   avg={avg_dh*1e6:.1f} us, med={med_dh*1e6:.1f} us, min={min_dh*1e6:.1f} us, max={max_dh*1e6:.1f} us")
        print(f"    +GEMM overlap: avg={avg_dho*1e6:.1f} us, med={med_dho*1e6:.1f} us, min={min_dho*1e6:.1f} us, max={max_dho*1e6:.1f} us")
        print(f"  Dispatch (single kernel, no hook split):")
        print(f"    avg={avg_ds*1e6:.1f} us, med={med_ds*1e6:.1f} us, min={min_ds*1e6:.1f} us, max={max_ds*1e6:.1f} us")
        print(f"    bandwidth: {total_dispatch_bytes / 1e9 / avg_ds:.2f} GB/s")
        print(f"  Full dispatch+combine (hook mode):")
        print(f"    no overlap:   avg={avg_fh*1e6:.1f} us, med={med_fh*1e6:.1f} us, min={min_fh*1e6:.1f} us, max={max_fh*1e6:.1f} us")
        print(f"    +GEMM overlap: avg={avg_fho*1e6:.1f} us, med={med_fho*1e6:.1f} us, min={min_fho*1e6:.1f} us, max={max_fho*1e6:.1f} us")
        print(flush=True)

    # =================================================================
    # Benchmark 3: torch.distributed.all_reduce baselines
    # =================================================================
    allreduce_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def torch_allreduce_bf16():
        dist.all_reduce(allreduce_bf16, group=group)

    avg_ar, min_ar, max_ar, med_ar = bench_fn(torch_allreduce_bf16, args.num_warmups, args.num_iters)
    allreduce_bytes = num_tokens * hidden * 2

    allreduce_fp32 = torch.randn((num_tokens, hidden), dtype=torch.float32, device='cuda')

    def torch_allreduce_fp32():
        dist.all_reduce(allreduce_fp32, group=group)

    avg_ar32, min_ar32, max_ar32, med_ar32 = bench_fn(torch_allreduce_fp32, args.num_warmups, args.num_iters)
    allreduce_fp32_bytes = num_tokens * hidden * 4

    # Expert-output-sized allreduce (fairer comparison)
    avg_tokens_per_expert = max(1, round(num_tokens * num_topk / num_experts))
    expert_output_tokens = num_local_experts * avg_tokens_per_expert
    expert_output = torch.randn((expert_output_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def torch_allreduce_expert():
        dist.all_reduce(expert_output, group=group)

    avg_are, min_are, max_are, med_are = bench_fn(torch_allreduce_expert, args.num_warmups, args.num_iters)
    expert_bytes = expert_output_tokens * hidden * 2

    if rank == 0:
        print(f"\n--- torch.distributed.all_reduce ---")
        print(f"  BF16 [{num_tokens}, {hidden}]:")
        print(f"    avg={avg_ar*1e6:.1f} us, med={med_ar*1e6:.1f} us, min={min_ar*1e6:.1f} us, max={max_ar*1e6:.1f} us")
        print(f"    data size: {allreduce_bytes / 1e6:.2f} MB, bandwidth: {allreduce_bytes / 1e9 / avg_ar:.2f} GB/s")
        print(f"  FP32 [{num_tokens}, {hidden}]:")
        print(f"    avg={avg_ar32*1e6:.1f} us, med={med_ar32*1e6:.1f} us, min={min_ar32*1e6:.1f} us, max={max_ar32*1e6:.1f} us")
        print(f"    data size: {allreduce_fp32_bytes / 1e6:.2f} MB, bandwidth: {allreduce_fp32_bytes / 1e9 / avg_ar32:.2f} GB/s")
        print(f"  BF16 [{expert_output_tokens}, {hidden}] (expert output size):")
        print(f"    avg={avg_are*1e6:.1f} us, med={med_are*1e6:.1f} us, min={min_are*1e6:.1f} us, max={max_are*1e6:.1f} us")
        print(f"    data size: {expert_bytes / 1e6:.2f} MB, bandwidth: {expert_bytes / 1e9 / avg_are:.2f} GB/s")
        print(flush=True)

    # ===== Summary =====
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"SUMMARY (median latency, us)")
        print(f"{'='*70}")
        print(f"  DeepEP dispatch (hook, no overlap):    {med_dh*1e6:>8.1f}")
        print(f"  DeepEP dispatch (hook, +GEMM overlap): {med_dho*1e6:>8.1f}")
        print(f"  DeepEP dispatch (single kernel):       {med_ds*1e6:>8.1f}")
        print(f"  DeepEP full d+c (hook, no overlap):    {med_fh*1e6:>8.1f}")
        print(f"  DeepEP full d+c (hook, +GEMM overlap): {med_fho*1e6:>8.1f}")
        print(f"  all_reduce BF16 [{num_tokens}x{hidden}]:        {med_ar*1e6:>8.1f}")
        print(f"  all_reduce FP32 [{num_tokens}x{hidden}]:        {med_ar32*1e6:>8.1f}")
        print(f"  all_reduce BF16 [{expert_output_tokens}x{hidden}]:       {med_are*1e6:>8.1f}")
        print(f"{'='*70}")
        print(flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark DeepEP LL vs all_reduce (sglang defaults)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens per rank (default: 128)')
    parser.add_argument('--num-warmups', type=int, default=50, help='Warmup iterations (default: 50)')
    parser.add_argument('--num-iters', type=int, default=100, help='Benchmark iterations (default: 100)')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes / GPUs (default: 8)')
    args = parser.parse_args()

    torch.multiprocessing.spawn(run_benchmark, args=(args.num_processes, args), nprocs=args.num_processes)
