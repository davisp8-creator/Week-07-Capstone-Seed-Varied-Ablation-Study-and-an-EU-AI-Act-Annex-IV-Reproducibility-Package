# Running the notebook on a Colab T4 GPU

This produces the T4 results, tagged `t4gpu`, and at the end offers to write a **local CPU vs Colab T4
comparison** into `README.md` and push it to GitHub with a fine-grained token. The local CPU results
(`results/*_cpu.*`) are never overwritten; they become the left-hand column of the comparison.

## 1. One-time: create a fine-grained token

GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token:

- **Repository access:** *Only select repositories* → this Week-07 repo
- **Permissions:** Repository permissions → **Contents: Read and write** (nothing else)
- **Expiration:** short (7–30 days); revoke it when the assignment is done

Optional, recommended: in Colab open the Secrets panel, add a secret named `GITHUB_TOKEN` with the token as its
value, and turn on *Notebook access*. The notebook then uses it automatically and never asks you to paste it.
Without the secret, you get a hidden prompt at the end (the token is not echoed or saved).

## 2. Runtime and notebook

**Runtime → Change runtime type → T4 GPU → Save.** Upload `Assignment_07_PE_Ablation_Colab.ipynb` (File → Upload notebook,
or open it from GitHub). The notebook's first code cell, **Colab Bootstrap**, clones this repo once and changes into it, so
there is nothing to type. It is safe to re-run: when `env_setup.py` already exists in the working directory it does nothing,
so **Run all** never clones twice or nests folders. The Bootstrap cell exists only when the notebook was built in a clone that
has a git remote named `origin`. This repo has no remote yet, so the clone URL is unset (`<REPO_URL>`): create the GitHub repo,
add the remote, and rebuild with `python scripts/build_notebook.py`.

Do not add your own `!git clone` / `%cd` cell: an unguarded one clones again on every Run all and descends one folder deeper
each time. If that already happened, use **Runtime → Disconnect and delete runtime**, then reconnect and Run all.

## 3. Run all

**Runtime → Run all.** Step 1 should print:

```text
[setup] host=colab-t4 (Colab (T4 GPU))  platform=colab  python=3.12.x  torch=2.x.x+cu12x  device=cuda  gpu=Tesla T4 (14.7 GB, sm_75)  results tag='t4gpu'
```

Check for **`host=colab-t4`**, **`gpu=Tesla T4`** and **`results tag='t4gpu'`**. If you see:

| Message | Meaning | Fix |
|---|---|---|
| `Colab is on a CPU-only runtime` | Step 2 was skipped | Change runtime type to T4 GPU, then Run all again |
| `nvidia-smi sees a GPU ... but torch cannot use it` | A CPU torch was installed over Colab's CUDA build | Runtime → Disconnect and delete runtime, then run again without installing anything |
| `GPU is <name>, not a T4` | Colab assigned a different GPU | Fine; results are tagged `gpu` so they are not mistaken for T4 numbers |
| `not inside a git clone` | The Colab Bootstrap cell was deleted or skipped | Restore it (rebuild with `python scripts/build_notebook.py`), or Disconnect and delete runtime, then Run all |

Don't `pip install -r requirements.txt` on Colab. It pins the local CPU torch and would hide the T4 from torch.

Training is the long step: 20 runs (4 arms x 5 seeds). The local CPU needs about 50 minutes; the T4 time is not measured
yet, so quote the figure from your own session's `manifest_t4gpu.json`, not an estimate. Progress is saved to
`results/runs_t4gpu.csv` after every run, so a disconnected session resumes where it stopped when you Run all again
(set `ABLATE_FRESH=1` to retrain everything).

## 4. The end-of-run prompts (Step 11)

```text
Write these results into README.md (local CPU vs Colab T4 comparison)? [y/N] y
[publish] README.md updated (results section only)
Commit and push README.md + results/*_t4gpu* to GitHub? [y/N] y
[publish] GitHub token (hidden; blank = use existing git credentials): ........
[publish] pulling origin/main ...
[publish] pushing to origin/main ...
```

What the push does, in order:

1. Commits the `*_t4gpu.*` result files (JSON, manifest, CSVs, markdown, plots).
2. Pulls `origin/main` with a rebase, so a newer CPU run pushed from your machine comes in here.
3. Re-renders the README results section from **all** results now in the clone and commits it.
4. Pushes.

The token is sent per command as an HTTP auth header. It is never printed, never written to `.git/config`, never added to
the remote URL, and never saved in the notebook. Answer **N** to either question to skip; nothing is pushed without your
second yes.

## 5. If you skipped the push

- **Later from Colab** (same session): `!python scripts/update_readme.py --push`
- **Re-render only, no git:** `python scripts/update_readme.py`

## What to expect from the comparison

Accuracy should agree between machines up to floating-point order (different kernels, so the last digits can differ).
Latency and memory are what the hardware changes. These models are small, so a T4 will not always beat a modern CPU on
latency: kernel-launch overhead dominates. That is a finding about this workload, not a bug.
