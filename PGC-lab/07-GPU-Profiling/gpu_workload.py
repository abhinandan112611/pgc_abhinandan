"""
Experiment 7: Profiling and GPU Utilization
-------------------------------------------
Workload: a stream of batches. Each batch is copied host -> GPU, processed
by a few kernels (GEMM + element-wise + reduction) and the result is copied
back.

  --mode baseline        pageable host memory, one stream, blocking copies
  --mode optimized       pinned host memory, two CUDA streams (copy + compute),
                         double buffering so copy of batch i+1 overlaps compute
                         of batch i

Examples:
  python gpu_workload.py --mode baseline --profile
  python gpu_workload.py --mode optimized --profile
"""
import argparse
import math
import time

import torch


# ----------------------------------------------------------------------------
# Workload pieces
# ----------------------------------------------------------------------------
def make_weights(n):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(n, n, generator=g) / math.sqrt(n)).cuda()


def make_host_batches(rows, cols, pinned, unique=4):
    """A few host batches that are re-used in a cycle (saves host RAM)."""
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(unique):
        t = torch.randn(rows, cols, generator=g)
        batches.append(t.pin_memory() if pinned else t)
    return batches


def compute(x, w, reps):
    """GPU work for one batch."""
    y = x
    for _ in range(reps):
        y = y @ w                            # GEMM             -> expected compute-bound
        y = torch.relu(y)                    # element-wise     -> expected memory-bound
        y = y.mul_(0.5).add_(1.0)            # element-wise     -> expected memory-bound
    return y.sum(dim=1)                      # reduction        -> expected memory-bound


# ----------------------------------------------------------------------------
# Version 1: BASELINE (pageable memory, default stream, blocking copies)
# ----------------------------------------------------------------------------
def run_baseline(host, w, reps, nbatches):
    results = []
    for i in range(nbatches):
        x = host[i % len(host)].cuda()   # blocking H2D copy from pageable memory
        r = compute(x, w, reps)          # kernels
        results.append(r.cpu())          # blocking D2H copy (CPU waits for GPU)
    torch.cuda.synchronize()
    return torch.stack(results)


# ----------------------------------------------------------------------------
# Version 2: OPTIMIZED (pinned memory + two streams + double buffering)
# ----------------------------------------------------------------------------
def run_optimized(host, w, reps, nbatches):
    rows, cols = host[0].shape
    copy_s = torch.cuda.Stream()                       # stream 1: data transfer
    comp_s = torch.cuda.Stream()                       # stream 2: computation

    buf = [torch.empty(rows, cols, device="cuda") for _ in range(2)]
    copied = [torch.cuda.Event() for _ in range(2)]    # "buffer k is filled"
    freed = [torch.cuda.Event() for _ in range(2)]     # "buffer k can be reused"
    out_host = torch.empty(nbatches, rows).pin_memory()

    torch.cuda.synchronize()
    for e in freed:
        e.record(comp_s)

    def issue_copy(i):
        k = i % 2
        copy_s.wait_event(freed[k])                    # do not overwrite a busy buffer
        with torch.cuda.stream(copy_s):
            buf[k].copy_(host[i % len(host)], non_blocking=True)   # async H2D
            copied[k].record(copy_s)

    issue_copy(0)
    for i in range(nbatches):
        k = i % 2
        if i + 1 < nbatches:
            issue_copy(i + 1)                                        # prefetch NEXT batch
        comp_s.wait_event(copied[k])                                 # wait for THIS batch only
        with torch.cuda.stream(comp_s):
            r = compute(buf[k], w, reps)
            freed[k].record(comp_s)                                  # buf[k] free after kernels
            out_host[i].copy_(r, non_blocking=True)                  # async D2H into pinned
    torch.cuda.synchronize()
    return out_host


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--rows", type=int, default=16384)   # 16384 x 1024 floats = 64 MB
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-batches", type=int, default=10)
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - check driver / PyTorch build.")

    pinned = a.mode == "optimized"
    fn = run_optimized if pinned else run_baseline
    host = make_host_batches(a.rows, a.cols, pinned)
    w = make_weights(a.cols)

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Mode            : {a.mode}   batch size: {a.rows * a.cols * 4 / 1e6:.0f} MB")

    fn(host, w, a.reps, 3)                             # warm-up (CUDA init, cuBLAS, allocator)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    res = fn(host, w, a.reps, a.batches)               # timed run
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(f"Batches    : {a.batches}")
    print(f"Total time : {dt:.3f} s")
    print(f"Per batch  : {dt / a.batches * 1000:.2f} ms")
    print(f"Throughput : {a.batches / dt:.1f} batches/s")
    print(f"Checksum   : {res.double().sum().item():.6e}")

    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn(host, w, a.reps, a.profile_batches)
            torch.cuda.synchronize()
        name = f"trace_{a.mode}.json"
        prof.export_chrome_trace(name)
        print(f"\nTrace saved: {name}")
        for key in ("cuda_time_total", "device_time_total"):
            try:
                print(prof.key_averages().table(sort_by=key, row_limit=10))
                break
            except Exception:
                continue


if __name__ == "__main__":
    main()
