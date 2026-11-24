"""
Experiment 7: read PyTorch profiler traces and report GPU utilization.

Usage:
  python analyze_trace.py trace_baseline.json trace_optimized.json
"""
import collections
import json
import sys

GAP_US = 50                                   # idle gap threshold (microseconds)
COMPUTE_HINTS = ("gemm", "sgemm", "cutlass", "cublas", "conv", "wmma")


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def overlap(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def label(name):
    low = name.lower()
    return "compute-bound" if any(h in low for h in COMPUTE_HINTS) else "memory-bound"


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data

    kern, copy = [], []
    knames = collections.defaultdict(lambda: [0, 0.0])
    cnames = collections.defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, s, d = e.get("cat", ""), e["ts"], e.get("dur", 0)
        if cat == "kernel":
            kern.append([s, s + d])
            knames[e["name"][:60]][0] += 1
            knames[e["name"][:60]][1] += d
        elif cat in ("gpu_memcpy", "gpu_memset"):
            copy.append([s, s + d])
            cnames[e["name"][:60]][0] += 1
            cnames[e["name"][:60]][1] += d

    if not kern:
        print(f"{path}: no GPU kernels found in trace")
        return None

    k, c = merge(kern), merge(copy)
    start = min(k[0][0], c[0][0] if c else k[0][0])
    end = max(k[-1][1], c[-1][1] if c else k[-1][1])
    span = end - start
    busy = total(k)
    gaps = [k[i + 1][0] - k[i][1] for i in range(len(k) - 1)]
    big = [g for g in gaps if g > GAP_US]

    print("=" * 72)
    print(path)
    print("=" * 72)
    print(f"Profiled window            : {span / 1000:9.2f} ms")
    print(f"Kernel busy time           : {busy / 1000:9.2f} ms")
    print(f"GPU utilization (busy/span): {100 * busy / span:9.1f} %")
    print(f"Copy (memcpy) busy time    : {total(c) / 1000:9.2f} ms")
    print(f"Copy overlapped with kernel: {overlap(k, c) / 1000:9.2f} ms")
    print(f"Idle gaps > {GAP_US} us        : {len(big):9d}   "
          f"(total {sum(big) / 1000:.2f} ms)")

    print("\nKernels (label rule: GEMM/cuBLAS -> compute-bound, else memory-bound)")
    print(f"{'calls':>6} {'total ms':>10} {'label':<14} name")
    for n, (cnt, dur) in sorted(knames.items(), key=lambda x: -x[1][1]):
        print(f"{cnt:6d} {dur / 1000:10.2f} {label(n):<14} {n}")

    print("\nMemory copies")
    print(f"{'calls':>6} {'total ms':>10} name")
    for n, (cnt, dur) in sorted(cnames.items(), key=lambda x: -x[1][1]):
        print(f"{cnt:6d} {dur / 1000:10.2f} {n}")
    print()
    return span, busy, total(c), overlap(k, c), len(big)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python analyze_trace.py trace_*.json")
    rows = []
    for p in sys.argv[1:]:
        r = analyze(p)
        if r:
            rows.append((p, r))
    if len(rows) > 1:
        print("SUMMARY")
        print(f"{'file':<24}{'window ms':>10}{'util %':>8}{'overlap ms':>12}{'gaps':>6}")
        for p, (span, busy, _, ov, g) in rows:
            print(f"{p:<24}{span / 1000:10.1f}{100 * busy / span:8.1f}"
                  f"{ov / 1000:12.1f}{g:6d}")


if __name__ == "__main__":
    main()
