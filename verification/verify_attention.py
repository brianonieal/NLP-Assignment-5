# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script to check
# the CausalSelfAttention code in Assignment5/gpt/model.py. I directed the work and reviewed it.
"""Checks the causal self-attention implementation in Assignment5/gpt/model.py.

1. Output matches torch.nn.functional.scaled_dot_product_attention(is_causal=True).
2. Changing tokens after position t leaves outputs at positions <= t unchanged.
3. Attention rows are proper distributions with zero weight on future positions.
4. A few hundred training steps on a small synthetic language lower the loss.
Run from the repo root: python verification/verify_attention.py
"""
import math, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Assignment5"))
from gpt.model import GPT, CausalSelfAttention
from gpt.utils import CfgNode as CN

torch.manual_seed(0)

def cfg(n_embd=48, n_head=3, block_size=32):
    c = CN(); c.n_embd, c.n_head, c.block_size = n_embd, n_head, block_size
    c.attn_pdrop = c.resid_pdrop = 0.0
    return c

# 1. match the reference implementation
c = cfg(); attn = CausalSelfAttention(c).eval()
x = torch.randn(4, 20, c.n_embd)
with torch.no_grad():
    ours = attn(x)
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split(C, dim=2)
    q, k, v = [t.view(B, T, c.n_head, C // c.n_head).transpose(1, 2) for t in (q, k, v)]
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.c_proj(ref.transpose(1, 2).contiguous().view(B, T, C))
err = (ours - ref).abs().max().item()
print(f"1. max |ours - reference| = {err:.2e}")
assert err < 1e-5

# 2. no information flows from the future
t = 9
x2 = x.clone(); x2[:, t + 1:, :] = torch.randn_like(x2[:, t + 1:, :])
with torch.no_grad():
    y1, y2 = attn(x), attn(x2)
past = (y1[:, :t + 1] - y2[:, :t + 1]).abs().max().item()
future = (y1[:, t + 1:] - y2[:, t + 1:]).abs().max().item()
print(f"2. change at positions > {t}: past outputs moved {past:.2e}, later outputs moved {future:.2e}")
assert past < 1e-6 and future > 1e-3

# 3. attention weights: rows sum to 1, upper triangle is 0
with torch.no_grad():
    q, k, v = attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T, c.n_head, -1).transpose(1, 2); k = k.view(B, T, c.n_head, -1).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    att = F.softmax(att.masked_fill(attn.causal_mask[:, :, :T, :T] == 0, float('-inf')), dim=-1)
upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
print(f"3. row sums in [{att.sum(-1).min():.6f}, {att.sum(-1).max():.6f}], max future weight {att[..., upper].abs().max():.1e}")
assert torch.allclose(att.sum(-1), torch.ones(1)) and att[..., upper].abs().max() == 0
print(f"   causal_mask buffer shape {tuple(attn.causal_mask.shape)}, lower-triangular: {bool(torch.equal(attn.causal_mask[0,0], torch.tril(torch.ones(c.block_size, c.block_size))))}")

# 4. the full GPT learns a simple next-token rule (token i is followed by i+1 mod V)
V = 50
mc = GPT.get_default_config(); mc.model_type = "gpt-nano"; mc.vocab_size = V; mc.block_size = 32
model = GPT(mc)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
def batch(b=64, T=32):
    start = torch.randint(0, V, (b, 1))
    return (start + torch.arange(T)) % V
losses = []
for step in range(300):
    ids = batch(); mask = torch.ones_like(ids, dtype=torch.float32)
    _, loss = model(ids, ids, mask)
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
print(f"4. synthetic LM loss: step 0 {losses[0]:.3f} (ln V = {math.log(V):.3f}), step 299 {losses[-1]:.3f}")
assert losses[-1] < 0.1 * losses[0]
print("all checks passed")
