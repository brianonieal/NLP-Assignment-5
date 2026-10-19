# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script to check
# the Section 2, 3, and 4 answers in Onieal_Assignment5.pdf. I directed the work and reviewed it.
"""Numerical checks for the written answers.

Section 2   logit-bias recovery of the full distribution from a top-2 API
Section 3.2 BPTT gradients of the identity-activation RNN versus PyTorch autograd
Section 4.1 operation counts of one self-attention layer
Section 4.2 masking one key position for every query
Section 4.3 PV = U (S (R^T V)) when P = U S R^T has rank k
Section 4.4 PV = Q (C K)^T (D V) with C = D = R^T
Run: python verification/verify_theory.py
"""
import torch
torch.manual_seed(641)
torch.set_default_dtype(torch.float64)

# ---------------- Section 2: logit bias ----------------
V = 1000
f = torch.randn(V) * 3                      # hidden logits
true_p = torch.softmax(f, 0)
calls = 0
def api(bias=None):
    """Returns the top-2 (token, probability) pairs after adding logit bias."""
    global calls; calls += 1
    logits = f.clone()
    if bias:
        for tok, b in bias.items(): logits[tok] += b
    p = torch.softmax(logits, 0)
    top = torch.topk(p, 2)
    return {int(i): float(pp) for i, pp in zip(top.indices, top.values)}

out0 = api()
v_star = max(out0, key=out0.get)
p_star = out0[v_star]
rec = torch.zeros(V); rec[v_star] = p_star
for v in range(V):
    if v == v_star: continue
    if v in out0:                    # the runner-up is already visible
        rec[v] = out0[v]; continue
    b = 40.0                         # large enough to lift v into the top 2
    out = api({v: b})
    assert v in out and v_star in out
    ratio_formula = p_star * torch.exp(torch.tensor(-b)) * out[v] / out[v_star]
    rec[v] = ratio_formula
print(f"S2  |V|={V}, API calls={calls} (limit |V|+1={V+1}), max |p_hat - p| = {(rec-true_p).abs().max():.2e}, sum p_hat = {rec.sum():.12f}")
# the hint's Z-based formula, needing only p(v*; b)
v = int(torch.argsort(f)[V // 2]); b = 5.0
out = api({v: b})
z_formula = (p_star / out[v_star] - 1) / (torch.exp(torch.tensor(b)) - 1)
print(f"S2  Z-formula for one token: {float(z_formula):.6e} vs true {float(true_p[v]):.6e}")

# ---------------- Section 3.2: RNN BPTT ----------------
T = 3
Whx = torch.randn(4, 3, requires_grad=True)
Whh = (0.5 * torch.randn(4, 4)).requires_grad_(True)
Wyh = torch.randn(2, 4, requires_grad=True)
xs = [torch.randn(3) for _ in range(T)]
ys = [torch.nn.functional.one_hot(torch.randint(0, 2, ()), 2).double() for _ in range(T)]
h0 = torch.randn(4)                          # nonzero so the h_0 terms are tested

hs, os_, yh = [h0], [], []
for t in range(T):
    z = Whh @ hs[-1] + Whx @ xs[t]
    h = z                                    # psi = identity
    o = Wyh @ h
    hs.append(h); os_.append(o); yh.append(torch.softmax(o, 0))
Jt = [-(ys[t] * torch.log(yh[t])).sum() for t in range(T)]
J = sum(Jt)

# 3.2.1 dJ_t/do_t = yhat_t - y_t
for t in range(T):
    g = torch.autograd.grad(Jt[t], os_[t], retain_graph=True)[0]
    assert torch.allclose(g, (yh[t] - ys[t]).detach())
print("3.2.1 dJ_t/do_t = yhat_t - y_t matches autograd for t=1,2,3")

g_o = [(yh[t] - ys[t]).detach() for t in range(T)]
Wh, Wy = Whh.detach(), Wyh.detach()
# 3.2.2 dJ_t/dh_i = (Whh^T)^(t-i) Wyh^T g_o_t
for t in range(T):
    for i in range(t + 1):
        auto = torch.autograd.grad(Jt[t], hs[i + 1], retain_graph=True)[0]
        mine = torch.linalg.matrix_power(Wh.T, t - i) @ Wy.T @ g_o[t]
        assert torch.allclose(auto, mine)
print("3.2.2 dJ_t/dh_i = (W_hh^T)^(t-i) W_yh^T (yhat_t - y_t) matches for all 6 (t, i) pairs")

# 3.2.3 dJ_t/dWhh = sum_{i<=t} g_h_i h_{i-1}^T
hs_d = [h.detach() for h in hs]
gW = []
for t in range(T):
    auto = torch.autograd.grad(Jt[t], Whh, retain_graph=True)[0]
    mine = sum(torch.outer(torch.linalg.matrix_power(Wh.T, t - i) @ Wy.T @ g_o[t], hs_d[i]) for i in range(t + 1))
    assert torch.allclose(auto, mine); gW.append(mine)
print("3.2.3 dJ_t/dW_hh = sum_i g_h_i h_{i-1}^T matches for t=1,2,3")

# 3.2.4 dJ/dWhh = sum_t g_Whh_t
auto = torch.autograd.grad(J, Whh)[0]
assert torch.allclose(auto, sum(gW))
print(f"3.2.4 dJ/dW_hh = sum_t g_W_hh,t matches (max abs diff {(auto - sum(gW)).abs().max():.1e})")
# the same gradient from one backward sweep: delta_i = dJ/dh_i summed over all J_t
delta = [None] * (T + 1)
delta[T] = Wy.T @ g_o[T - 1]
for i in range(T - 1, 0, -1):
    delta[i] = Wy.T @ g_o[i - 1] + Wh.T @ delta[i + 1]
sweep = sum(torch.outer(delta[i], hs_d[i - 1]) for i in range(1, T + 1))
assert torch.allclose(auto, sweep)
print(f"3.2.4 one-sweep form sum_i delta_i h_(i-1)^T matches (max abs diff {(auto - sweep).abs().max():.1e})")

# ---------------- Section 4.1: operation counts ----------------
def mults(n, h):
    proj = 3 * n * h * h        # XW_q, XW_k, XW_v
    scores = n * n * h          # QK^T
    mix = n * n * h             # A V
    return proj, scores, mix
for n in (128, 256, 512, 1024):
    p, s, m = mults(n, 64)
    print(f"4.1 n={n:5d} h=64: projections {p:>12,d}  QK^T {s:>12,d}  AV {m:>12,d}  total {p+s+m:>13,d}")

# ---------------- Sections 4.3 and 4.4 ----------------
n, h, k = 300, 64, 12
Q = torch.randn(n, h); K = torch.randn(n, h); Vv = torch.randn(n, h)
A = torch.randn(n, k); B = torch.randn(k, n)
P = torch.softmax(A @ B, 1)                      # not low rank in general...
U0, S0, R0 = torch.linalg.svd(P)
P = U0[:, :k] @ torch.diag(S0[:k]) @ R0[:k, :]   # ...so force rank k for the 4.3 test
U, S, Rt = torch.linalg.svd(P)
U, S, Rt = U[:, :k], S[:k], Rt[:k, :]
lhs = P @ Vv
rhs = U @ (S[:, None] * (Rt @ Vv))
print(f"4.3 rank(P)={torch.linalg.matrix_rank(P)}, max |PV - U(S(R^T V))| = {(lhs - rhs).abs().max():.2e}")

Qk = torch.randn(n, k) @ torch.randn(k, h)       # makes P = Q K^T rank k
P2 = Qk @ K.T
U, S, Rt = torch.linalg.svd(P2)
C = D = Rt[:k, :]                                 # k x n
lhs = P2 @ Vv
rhs = Qk @ ((C @ K).T @ (D @ Vv))
print(f"4.4 rank(QK^T)={torch.linalg.matrix_rank(P2)}, max |PV - Q(CK)^T(DV)| = {(lhs - rhs).abs().max():.2e} (relative {((lhs-rhs).norm()/lhs.norm()):.1e})")
# 4.4 with the 1/sqrt(h) scaling kept: fold the scalar into C
hh = Qk.shape[1]
P3 = Qk @ K.T / hh ** 0.5
C3 = Rt[:k, :] / hh ** 0.5
lhs = P3 @ Vv
rhs = (Qk @ (C3 @ K).T) @ (Rt[:k, :] @ Vv)
print(f"4.4 with scaling, C = R^T/sqrt(h), D = R^T: relative error {((lhs-rhs).norm()/lhs.norm()):.1e}")
assert (lhs - rhs).norm() / lhs.norm() < 1e-10

# Section 2 again with the Z-formula for every token, a small bias (b = 2), and log-probabilities
b = 2.0
rec2 = torch.zeros(V); rec2[v_star] = p_star
calls_before = calls
for v in range(V):
    if v == v_star: continue
    out = api({v: b})
    assert v_star in out                  # v* always stays in the top 2 when one other token is biased
    rec2[v] = (p_star / out[v_star] - 1) / (torch.exp(torch.tensor(b)) - 1)
print(f"S2  Z-formula, b={b}: {calls - calls_before} biased calls, max |p_hat - p| = {(rec2-true_p).abs().max():.2e}, max relative error = {((rec2-true_p).abs()/true_p).max():.2e}")
# ---------------- Section 4.2: masking position 3 ----------------
n2, h2 = 6, 8
X = torch.randn(n2, h2); Wq, Wk, Wv = (torch.randn(h2, h2) for _ in range(3))
def masked_attention(X):
    S = (X @ Wq) @ (X @ Wk).T / h2 ** 0.5
    M = torch.zeros(n2, n2); M[:, 3] = float("-inf")    # every query loses key 3
    A = torch.softmax(S + M, dim=-1)
    return A, A @ (X @ Wv)
A, out = masked_attention(X)
X2 = X.clone(); X2[3] = torch.randn(h2)                  # change token 3 only
A2, out2 = masked_attention(X2)
others = [t for t in range(n2) if t != 3]
print(f"4.2 column 3 weights max {A[:, 3].abs().max():.1e}, row sums in [{A.sum(-1).min():.12f}, {A.sum(-1).max():.12f}], "
      f"rows != 3 moved {(out[others] - out2[others]).abs().max():.1e} when x_3 changed, row 3 moved {(out[3] - out2[3]).abs().max():.2f}")
assert A[:, 3].abs().max() == 0 and torch.allclose(A.sum(-1), torch.ones(n2))
assert torch.allclose(out[others], out2[others])
print("all checks passed")
