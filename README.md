# Assignment 5: RNNs and Transformers/Self-Attention

EN.705.641.81 Natural Language Processing: Self-Supervised Models
Johns Hopkins University Engineering for Professionals, Fall 2026

Brian W. Onieal

GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote the answers and derivations and filled in the code TODOs. It also wrote the helper and verification scripts and ran the training on my laptop GPU through the Claude app. I chose the scope and reviewed the work before submitting. Every `.py` file Claude wrote or changed says at its top what Claude did. The skeleton files with no header are unchanged.

## What's here

| File | What it is |
|---|---|
| `Onieal_Assignment5.pdf` | Written answers to Sections 1 to 5, with the computation graph, both plots, and the completions |
| `Onieal_Assignment5.docx` | The same answers in the Word version of the assignment (answers are in blue under each question) |
| `Assignment5/gpt/model.py` | `CausalSelfAttention` with the causal mask and the attention forward pass filled in |
| `Assignment5/gpt/trainer.py` | The skeleton's trainer plus a `micro_batches` option (see "Changes outside the TODOs") |
| `Assignment5/main.py` | The skeleton's entry point with `micro_batches = 2` |
| `Assignment5/data.py`, `Assignment5/generate.py`, `Assignment5/gpt/bpe.py`, `Assignment5/gpt/utils.py`, `Assignment5/requirements.txt` | Unchanged from the skeleton |
| `Assignment5/Assignment5.md` | The skeleton's setup notes, with the folder, environment, and assignment names updated |
| `Assignment5/loss.png`, `Assignment5/ppl.png` | The two plots `gpu_full_run` saved |
| `Assignment5/run_log.txt` | Console output of the reported run: the 50-step CPU and GPU comparison, the 20,000-step GPU run, and the four completions |
| `Assignment5/run_log_first_attempt.txt` | Console output of the first attempt, which trained to the end and then crashed in `plot()` on a broken Tk install |
| `Assignment5/curves.json` | Training loss and perplexity at every step, and validation loss and perplexity every 1,000 steps, behind the plots |
| `Assignment5/model.pth` | The saved checkpoint (best validation perplexity, step 20,000) that produced the completions |
| `Assignment5_code.zip` | The source files, since the assignment asks for a zip |
| `scripts/prefetch_wikitext.py` | Downloads the WikiText-103 parquet files into `data_cache/` (not committed, about 300 MB) |
| `scripts/run_main_local.py` | Runs `Assignment5/main.py` as written, reading WikiText-103 from `data_cache/` and saving plots with matplotlib's Agg backend |
| `verification/` | Scripts that check the written answers and the code, each with its output |
| `writeup/` | Markdown source of the Word document, the computation graph source, and the build and PDF export scripts |
| `Assignment5.docx`, `Assignment5.pdf` | The assignment, as shipped in the template (renamed) |

## Results

| Measurement | Value |
|---|---:|
| 50 training steps on the CPU (Intel Core Ultra 9 185H), reported run | 156.88 s |
| 50 training steps on the GPU (NVIDIA GeForce RTX 4050 Laptop, 6 GB), reported run | 10.65 s |
| GPU speedup over the CPU across five runs of the 50-step comparison | 7.8 to 14.7 times, median 8.4 |
| 20,000 training steps on the GPU | 4,254.87 s |
| Validation loss, step 0 and step 20,000 | 10.831 and 4.834 |
| Validation perplexity, step 0 and step 20,000 | 50,544.53 and 128.96 |

gpt-nano: 3 layers, 3 heads, 48-dimensional states, block size 32, batch size 256, learning rate 5e-4, 4,911,120 parameters (`main.py` prints 2.50M because its count leaves out the 2,412,336 in the output layer). Validation improved at every one of the 21 evaluations. The first attempt printed 107.50 s on the CPU and 10.69 s on the GPU for the same 50 steps. Its checkpoint has the same MD5 as `Assignment5/model.pth`, so the seeded run reproduces. Because the CPU time moved between the two September 29 runs, `verification/cpu_gpu_repeat.py` repeated the comparison three times on October 19: the CPU took 107.36 to 117.41 s and the GPU 13.77 to 13.98 s.

## How to run

From the repository root:

```
pip install -r Assignment5/requirements.txt
python scripts/prefetch_wikitext.py          # once, about 300 MB, into data_cache/
cd Assignment5
python ../scripts/run_main_local.py          # or python main.py, which downloads from the Hub
cd ..
python verification/verify_theory.py         # Sections 2, 3.2, 4.1 to 4.4
python verification/verify_attention.py      # the CausalSelfAttention code
python verification/verify_microbatch.py     # the micro_batches change
python verification/attention_scaling.py     # attention time against sequence length, CPU and GPU
python verification/gpu_memory_probe.py      # peak GPU memory for one batch of 256, whole and in chunks
python verification/cpu_gpu_repeat.py        # the 50-step CPU and GPU comparison, three times
```

## Where it ran

My laptop: Windows 11 Pro, Intel Core Ultra 9 185H (16 cores), NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Python 3.11.9, torch 2.1.0+cu121, transformers 4.37.1, datasets 2.17.1, matplotlib 3.8.3. The assignment's `sbatch.sh` is for a Slurm cluster, so it isn't used here.

## Changes outside the TODOs

- `Assignment5/gpt/trainer.py`: a `micro_batches` option splits each batch into equal chunks for the forward and backward pass. Each chunk's loss is weighted by its share of the batch, so the gradient matches the full batch. With `micro_batches = 1` it behaves like the skeleton. `Assignment5/main.py` sets it to 2 so a batch of 256 fits in 6 GB. `verification/gpu_memory_probe.py` measured why: one batch of 256 peaks at 6.62 GB, over the GPU's 6.44 GB, and a step takes 4,667 ms as memory spills over. As two chunks it peaks at 3.37 GB and takes 270 ms. `verification/verify_microbatch.py` checks the loss and gradient against one full batch. The trainer also prints a blank line after the elapsed time.
- `Assignment5/gpt/model.py`: the TODO asks for a tensor named `causal_mask`, and the provided lines after it read `casual_mask`, so both names point at the same tensor.
- `scripts/run_main_local.py` replaces `datasets.load_dataset` for the one WikiText-103 call in `Assignment5/data.py` and reads the same rows from local parquet files. My network kept resetting connections to huggingface.co partway through the download. It also saves the curves to `Assignment5/curves.json` before plotting.
