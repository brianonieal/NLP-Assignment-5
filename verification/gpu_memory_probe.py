# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Measures peak GPU memory and time per training step for gpt-nano at batch size 256.

This is the measurement behind micro_batches = 2 in Assignment5/main.py. It runs the Assignment 5 GPT on random
token ids with the same shapes main.py uses (batch 256, block size 32, GPT-2 vocabulary), once
as a single batch and once split into two and four chunks, with the same size-weighted loss that
Trainer.chunked_loss uses. Run on the GPU after training has finished:
python verification/gpu_memory_probe.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Assignment5"))
from gpt.model import GPT

torch.manual_seed(0)
cfg = GPT.get_default_config(); cfg.model_type = "gpt-nano"; cfg.vocab_size = 50257; cfg.block_size = 32
model = GPT(cfg).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
ids = torch.randint(0, 50256, (256, 32), device="cuda")
mask = torch.ones(256, 32, device="cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1e9:.2f} GB")

def step(chunks):
    opt.zero_grad(set_to_none=True)
    for i, m in zip(ids.chunk(chunks), mask.chunk(chunks)):
        _, loss = model(i, i, m)
        (loss * i.size(0) / ids.size(0)).backward()
    opt.step()

for chunks, n in ((2, 20), (4, 20), (1, 5)):
    for _ in range(2):
        step(chunks)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(n):
        step(chunks)
    torch.cuda.synchronize()
    print(f"batch 256 as {chunks} chunk(s): {1000 * (time.time() - t) / n:.0f} ms/step, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
