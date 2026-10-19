# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this helper. I directed
# the work and reviewed it.
"""Downloads the WikiText-103 (raw) parquet files, one file at a time, retrying on resets.

My network reset about half of all connections to huggingface.co, and load_dataset() in
datasets 2.17.1 makes dozens of requests without retrying, so load_data() in Assignment5/data.py
could not finish. The Hub's "wikitext" dataset is now served from these parquet files
(it redirects to Salesforce/wikitext), so the rows are the same ones load_data() reads.
Files go to data_cache/wikitext-103-raw-v1/ at the repo root (not committed).
"""
import json, os, time
from huggingface_hub import HfApi, hf_hub_download

REPO, CONFIG = "Salesforce/wikitext", "wikitext-103-raw-v1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")

def retry(fn, what, tries=30):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"{what}: attempt {attempt} failed: {type(e).__name__}", flush=True)
            time.sleep(2)
    raise SystemExit(f"{what}: gave up after {tries} attempts")

api = HfApi()
info = retry(lambda: api.dataset_info(REPO), "dataset_info")
sha = info.sha
files = retry(lambda: [f for f in api.list_repo_files(REPO, repo_type="dataset", revision=sha)
                       if f.startswith(CONFIG + "/") and f.endswith(".parquet")], "list files")
print(f"{REPO} @ {sha}: {files}", flush=True)
for f in files:
    path = retry(lambda: hf_hub_download(REPO, f, repo_type="dataset", revision=sha, local_dir=OUT), f)
    print(f"ok {f} {os.path.getsize(path)/1e6:.1f} MB", flush=True)
with open(os.path.join(OUT, CONFIG, "SOURCE.json"), "w") as fh:
    json.dump({"repo": REPO, "revision": sha, "files": files}, fh, indent=2)
print("PREFETCH_DONE", flush=True)
