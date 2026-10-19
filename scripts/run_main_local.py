# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this helper. I directed
# the work and reviewed it.
"""Runs Assignment5/main.py unchanged, with WikiText-103 read from the local parquet files.

It also sets matplotlib to the Agg backend and saves the per-step training curves to
Assignment5/curves.json just before Trainer.plot() draws them.

Run prefetch_wikitext.py first. This replaces datasets.load_dataset before main.py is
imported, only for the exact call Assignment5/data.py makes (path="wikitext",
name="wikitext-103-raw-v1"), and loads the same split from the downloaded parquet files
instead of the Hub. Every other line of main.py, data.py, and gpt/ runs as written.
Run from Assignment5/: python ../scripts/run_main_local.py
"""
import glob, json, os, runpy, sys
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Tk is broken in this machine's Python 3.11 install, and matplotlib's default Windows backend
# (TkAgg) needs it. Trainer.plot() only saves PNGs, so the non-GUI Agg backend is all it needs.
os.environ.setdefault("MPLBACKEND", "Agg")
import datasets

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOCAL = os.path.join(ROOT, "data_cache", "wikitext-103-raw-v1")
_original = datasets.load_dataset

def load_dataset(path=None, name=None, split=None, **kwargs):
    if path == "wikitext" and name == "wikitext-103-raw-v1":
        files = sorted(glob.glob(os.path.join(LOCAL, f"{split}-*.parquet")))
        assert files, f"no local parquet files for split {split!r} in {LOCAL}"
        print(f"(local) {split}: {[os.path.basename(f) for f in files]}", flush=True)
        return _original("parquet", data_files={split: files}, split=split)
    return _original(path, name=name, split=split, **kwargs)

datasets.load_dataset = load_dataset
code_dir = os.path.join(ROOT, "Assignment5")
sys.path.insert(0, code_dir)
os.chdir(code_dir)

# Save the per-step curves before plotting, so a plotting failure can't lose them.
from gpt import trainer as _trainer
_plot = _trainer.Trainer.plot
def plot_and_save(self):
    with open("curves.json", "w") as fh:
        json.dump({"train_loss": self.all_iter_train_loss, "train_ppl": self.all_iter_train_ppl,
                   "valid_loss": self.all_iter_valid_loss, "valid_ppl": self.all_iter_valid_ppl}, fh)
    print("saved curves.json", flush=True)
    _plot(self)
_trainer.Trainer.plot = plot_and_save
runpy.run_path(os.path.join(code_dir, "main.py"), run_name="__main__")
