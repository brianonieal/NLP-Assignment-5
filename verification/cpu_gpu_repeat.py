# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Repeats main.py's cpu_gpu_comparison (50 training steps on the CPU, then 50 on the GPU) three
times in one process, to see how much the printed times vary from run to run.

It calls main.run with the same settings main.py uses, after loading WikiText-103 from
data_cache/ the same way scripts/run_main_local.py does. It works from a temporary folder, so the
checkpoint each short run saves can't overwrite Assignment5/model.pth.
Run from the repo root: python verification/cpu_gpu_repeat.py
"""
import glob, io, os, re, statistics, sys, tempfile
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCAL = os.path.join(ROOT, "data_cache", "wikitext-103-raw-v1")
_original = datasets.load_dataset

def load_dataset(path=None, name=None, split=None, **kwargs):
    if path == "wikitext" and name == "wikitext-103-raw-v1":
        files = sorted(glob.glob(os.path.join(LOCAL, f"{split}-*.parquet")))
        assert files, f"no local parquet files for split {split!r} in {LOCAL}"
        return _original("parquet", data_files={split: files}, split=split)
    return _original(path, name=name, split=split, **kwargs)

datasets.load_dataset = load_dataset
sys.path.insert(0, os.path.join(ROOT, "Assignment5"))
os.chdir(os.path.join(ROOT, "Assignment5"))
import torch
from data import load_data, create_datasets, VOCAB_SIZE
from gpt.model import GPT
from main import run

cfg = GPT.get_default_config(); cfg.model_type = "gpt-nano"; cfg.vocab_size = VOCAB_SIZE; cfg.block_size = 32
m = GPT(cfg)
total = sum(p.numel() for p in m.parameters())
head = sum(p.numel() for p in m.lm_head.parameters())
print(f"parameters: {total:,} in all, {total - head:,} without lm_head, {head:,} in lm_head", flush=True)
print(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}, GPU {torch.cuda.get_device_name(0)}", flush=True)

train_data, dev_data = load_data()
train_d, dev_d = create_datasets(train_data, dev_data, block_size=32)
os.chdir(tempfile.mkdtemp(prefix="a5_repeat_"))

class Tee(io.TextIOBase):
    def __init__(self, out): self.out, self.buf = out, []
    def write(self, s): self.buf.append(s); return self.out.write(s)
    def flush(self): self.out.flush()

times = {"cpu": [], "cuda": []}
for rep in range(1, 4):
    for device in ("cpu", "cuda"):
        tee = Tee(sys.stdout); old = sys.stdout; sys.stdout = tee
        try:
            run(train_d, dev_d, max_iter=50, device=device, plot=False, sample=False)
        finally:
            sys.stdout = old
        t = float(re.findall(r"Trainer Time Elapsed: ([\d.]+)s", "".join(tee.buf))[-1])
        times[device].append(t)
        print(f"repeat {rep} {device}: {t:.2f} s", flush=True)

for device, ts in times.items():
    print(f"{device}: {', '.join(f'{t:.2f}' for t in ts)} s, median {statistics.median(ts):.2f} s", flush=True)
ratios = [c / g for c, g in zip(times["cpu"], times["cuda"])]
print(f"CPU/GPU per repeat: {', '.join(f'{r:.1f}' for r in ratios)}, median {statistics.median(ratios):.1f}", flush=True)
