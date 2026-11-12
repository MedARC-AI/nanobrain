# nanobrain

A collaborative (or competitive) challenge to find the best recipe for training brain MRI foundation models. Inspired by [nanoGPT](https://github.com/karpathy/nanogpt), [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt) and our sibling project [nanopath](https://github.com/MedARC-AI/nanopath).

```
/goal achieve the best brainmarks-smri open score, using only the provided fomo300k pretraining data and at most 1 hour on 1xH100.
```

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #neuro-fm)!**

## Leaderboard

<a href="https://labless.dev/nano-projects/nanobrain-smri">
  <img src="https://api.labless.dev/api/nano-projects/nanobrain-smri/plot.svg" alt="nanobrain progress plot">
</a>

| # | Name | open | classification | age | segmentation | Contributors |
|---|---|---:|---:|---:|---:|---|
| 1 | [screen_s2_a0](https://github.com/MedARC-AI/nanobrain/tree/valid/screen_s2_a0) | 0.556 | 0.327 | 0.945 | 0.755 | [@RyanKim17920](https://github.com/RyanKim17920) |
| 2 | [r12_nat_lvl_mf](https://github.com/MedARC-AI/nanobrain/tree/valid/r12_nat_lvl_mf) | 0.529 | 0.309 | 0.863 | 0.747 | [@mihirneal](https://github.com/mihirneal) |
| 3 | [hp1hr-lr1.5e-3-s0](https://github.com/MedARC-AI/nanobrain/tree/valid/hp1hr-lr1.5e-3-s0) | 0.508 | 0.299 | 0.858 | 0.694 | [@clane9](https://github.com/clane9) |
| 4 | [baseline](https://github.com/MedARC-AI/nanobrain/tree/ac1c1d8c04a6079dac666d07fdf846e36031e06a) | 0.475 | 0.260 | 0.836 | 0.667 | [@clane9](https://github.com/clane9) |

The target metric is the [brainmarks-smri](https://github.com/MedARC-AI/brainmarks-smri) `open` score, which is the mean over 11 tasks from 9 public MRI benchmark datasets. Sub-scores are means over each task type. Metrics are normalized so that 0 is ~chance and 1 is perfect: classification (2 AUROC - 1), age (R²) and segmentation (Dice). Scores are the median of 3 training seeds.

### Baselines

| # | Name | open | classification | age | segmentation |
|---|---|---:|---:|---:|---:|
| 1 | [Walnut](https://labless.dev/runs/run_sub_3b6f4d9185) | 0.580 | 0.343 | 0.958 | 0.801 |
| 2 | [Walnut 0.1](https://github.com/MedARC-AI/walnut-fomo26) | 0.556 | 0.332 | 0.923 | 0.760 |
| 3 | [nnFoundation ViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | 0.492 | 0.283 | 0.841 | 0.678 |
| 4 | [Neuro-JEPA](https://github.com/NYUMedML/Neuro-JEPA) | 0.477 | 0.325 | 0.886 | 0.510 |
| 5 | [NeuroVFM](https://github.com/MLNeurosurg/neurovfm) | 0.468 | 0.317 | 0.881 | 0.495 |
| 6 | [OpenMind SimCLR](https://huggingface.co/MIC-DKFZ/ResEncL-OpenMind-SimCLR) | 0.458 | 0.305 | 0.818 | 0.523 |
| 7 | [BrainDINO](https://github.com/mclwu22/BrainDINO) | 0.439 | 0.287 | 0.791 | 0.508 |
| 8 | [DINOv3](https://github.com/facebookresearch/dinov3) | 0.429 | 0.290 | 0.761 | 0.485 |
| 9 | [Handcrafted](https://github.com/MedARC-AI/brainmarks-smri/blob/d1eef0e91a7da5ce62aab15d1e7f437d89555d4b/src/brainmarks_smri/models/handcrafted.py) | 0.406 | 0.235 | 0.799 | 0.485 |
| 10 | [VoCo](https://github.com/Luffy03/VoCo) | 0.402 | 0.260 | 0.720 | 0.473 |
| 11 | [AMAES](https://github.com/asbjrnmunk/amaes) | 0.375 | 0.219 | 0.591 | 0.545 |
| 12 | [BrainIAC](https://github.com/AIM-KannLab/BrainIAC) | 0.329 | 0.176 | 0.579 | 0.471 |

Baselines are frozen reference models scored with the same [brainmarks-smri](https://github.com/MedARC-AI/brainmarks-smri) probes. Walnut trained for ~15 hours on 8×H100, outside the leaderboard budget.

## Rules

- Training gets 1 hour of wall time on 1xH100 (~3.6e18 theoretical FLOPs).
- Pretrain only on the FOMO300K data in brainmarks-smri.
- The score is the mean over the fixed 11 brainmarks-smri `open` tasks.
- No weights or teachers pretrained on brain MRI.
- Test-time tricks on the embeddings (combining layers, pooling strategies, test-time augmentation) are technically allowed but lose style points.

Anything else can change: data curation, model, objective, preprocessing, schedule, etc.

## Setup

```sh
git clone https://github.com/MedARC-AI/nanobrain && cd nanobrain
uv sync
```

Training and eval both read the data folder in `BRAINMARKS_SMRI_DATA`. On the MedARC cluster:

```sh
echo "BRAINMARKS_SMRI_DATA=/data/brainmarks-smri" >> .env
```

Elsewhere, download it from the gated Hugging Face repo [medarc/brainmarks-smri](https://huggingface.co/datasets/medarc/brainmarks-smri) (~850 GB). Accept its terms first, then:

```sh
uv run hf auth login
uv run hf download medarc/brainmarks-smri --repo-type dataset --local-dir data
echo "BRAINMARKS_SMRI_DATA=${PWD}/data" >> .env
```

## Train, evaluate, submit

```sh
uv run python scripts/submit.py --login-only
sbatch scripts/train.sbatch smoke notes="smoke test" time_limit=5
sbatch scripts/train.sbatch my-run notes="what changed and why"
```

`train.sbatch` trains for 1 hour (`time_limit`), then evaluates the last checkpoint on the brainmarks-smri `open` tasks and submits it, except smoke tests.

`submit.py` posts the score and your code to [labless](https://labless.dev/nano-projects/nanobrain-smri). Sign in once with `--login-only` before your first job. Remember to push your branch so others can check out your commit.

## Code

The baseline is a 3D ViT-B masked autoencoder trained on 87K MRI brain volumes (T1w, T2w, FLAIR, DWI) at 1mm isotropic resolution with an 8mm patch size and an effective masking ratio of ~80%.

| | |
|---|---|
| [`train.py`](src/nanobrain/train.py) | training loop |
| [`model.py`](src/nanobrain/model.py) | ViT MAE and the brainmarks-smri model contract |
| [`data.py`](src/nanobrain/data.py) | FOMO300K dataset and preprocessing |
| [`default_pretrain.yaml`](src/nanobrain/config/default_pretrain.yaml) | default config |
| [`select_images.py`](src/nanobrain/select_images.py) | pretraining image selection, writes `fomo300k_images.txt` |
| [`train.sbatch`](scripts/train.sbatch) | train, then evaluate |
| [`submit.py`](scripts/submit.py) | submit to the leaderboard |
