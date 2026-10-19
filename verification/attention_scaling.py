# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Times one single-head self-attention layer, softmax(Q K^T / sqrt(h)) V plus the three
projections, at growing sequence lengths n with h = 64, on the CPU and on the GPU.

It backs the Section 4.1 answer: the work grows as n^2 h, and the GPU runs the n^2 dot
products in parallel. Run: python verification/attention_scaling.py
"""
import math, time
import torch

torch.manual_seed(0)
h = 64

def layer(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = torch.softmax(Q @ K.T / math.sqrt(h), dim=-1)
    return A @ V

def bench(device, n, reps):
    X = torch.randn(n, h, device=device)
    W = [torch.randn(h, h, device=device) / math.sqrt(h) for _ in range(3)]
    for _ in range(2):
        layer(X, *W)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        layer(X, *W)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / reps

print(f"CPU threads: {torch.get_num_threads()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'n':>6} {'mult-adds (3nh^2 + 2n^2h)':>26} {'CPU ms':>9} {'GPU ms':>9}")
for n in (512, 1024, 2048, 4096, 8192):
    flops = 3 * n * h * h + 2 * n * n * h
    cpu = bench("cpu", n, 5)
    gpu = bench("cuda", n, 20) if torch.cuda.is_available() else float("nan")
    print(f"{n:>6} {flops:>26,d} {cpu:>9.2f} {gpu:>9.3f}")
