# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script to check the
# micro_batches change in Assignment5/gpt/trainer.py. I directed the work and reviewed it.
"""Checks that Trainer.chunked_loss gives the same loss and gradient as one full batch.

Uses a padded batch with uneven sequence lengths, like collate_fn produces, dropout off,
and an odd batch size so the chunks are unequal (the last batch of an epoch can be).
Run from the repo root: python verification/verify_microbatch.py
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Assignment5"))
from gpt.model import GPT
from gpt.trainer import Trainer
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

V, PAD = 500, 499
mc = GPT.get_default_config(); mc.model_type = "gpt-nano"; mc.vocab_size = V; mc.block_size = 32
mc.embd_pdrop = mc.resid_pdrop = mc.attn_pdrop = 0.0
model = GPT(mc)
lengths = torch.randint(2, 33, (37,))
ids = torch.full((37, 32), PAD)
for r, L in enumerate(lengths): ids[r, :L] = torch.randint(0, PAD, (L,))
mask = (ids != PAD).double()

def run(chunks):
    tc = Trainer.get_default_config(); tc.device = "cpu"; tc.micro_batches = chunks
    tr = Trainer(tc, model, None, None)
    model.zero_grad(set_to_none=True)
    loss = tr.chunked_loss(ids, ids, mask, backward=True)
    return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

l1, g1 = run(1)
for chunks in (2, 3):
    lc, gc = run(chunks)
    print(f"chunks={chunks}: |loss diff| = {abs(lc - l1):.2e}, max |grad diff| = {(gc - g1).abs().max():.2e} (grad scale {g1.abs().max():.2e})")
    assert abs(lc - l1) < 1e-12 and torch.allclose(gc, g1, atol=1e-12)
_, full = model(ids, ids, mask)
print(f"unchunked GPT.forward loss {full.item():.10f} vs chunked_loss(chunks=1) {l1:.10f}")
assert abs(full.item() - l1) < 1e-12
print("all checks passed")
