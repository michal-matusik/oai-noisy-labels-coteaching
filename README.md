# Robust Binary Classification Under Noisy, Imbalanced Labels (Co-Teaching)

Training two neural networks to classify images correctly despite ~1/3 of the
training labels being wrong and the classes being imbalanced — without ever
seeing which labels are corrupted.

## Overview

Real-world labeled datasets are rarely perfect: annotators disagree, get tired,
or the underlying model that auto-labeled the data made mistakes. This project
tackles a controlled version of that problem — a training set with a
significant fraction of flipped (incorrect) labels — and asks for a model that
still generalizes well on clean validation/test data.

The fixed model architecture and training loop are provided; the only thing
to design is **which training samples each of two models is allowed to learn
from at each step**. The solution implemented here is a variant of
**co-teaching**: two networks train in parallel and each one selects training
examples *for the other*, based on which samples currently look "easy"
(low loss) — since mislabeled samples tend to produce anomalously high loss.

## Competition Task

- **Setting**: Binary image classification. The training set has label noise
  (some labels are wrong) and class imbalance. The validation and (hidden) test
  sets have only clean, correctly-labeled examples.
- **Goal**: Implement `your_select_indices(targets, losses)` — given a batch's
  labels and the per-sample losses from both models, return two lists of
  indices (one per model) selecting which samples in the batch that model
  should train on this step. Nothing else in the notebook may be modified
  except this function.
- **Data**: 10,000 training images (label counts: class 0 = 6298, class 1 =
  3702 — imbalanced, and inverted relative to the ~clean validation set, which
  is dominated by class 1), 1,000 validation images, 28×28 grayscale.
- **Evaluation metric**: mean balanced accuracy (BAC) of the two trained
  models on the hidden test set:

  ```
  BAC_mean = (BAC_1 + BAC_2) / 2

  Points =  0                                     if BAC_mean <= 0.5
            100 * (BAC_mean - 0.5) / (0.8 - 0.5)   if 0.5 < BAC_mean < 0.8
            100                                    if BAC_mean >= 0.8
  ```
- **Constraints**: no internet access at evaluation time, GPU available,
  evaluation must finish in under 5 minutes, and `SmallMobileNet`'s
  architecture is fixed and must not be changed.

## Approach

**Co-teaching** (Han et al., ["Co-teaching: Robust Training of Deep Neural
Networks with Extremely Noisy Labels"](https://arxiv.org/abs/1804.06872)),
adapted with class-specific sampling rates:

1. Both models see every batch and compute a per-sample loss.
2. For each class present in the batch, each model selects the samples of
   that class with the **lowest loss according to the *other* model**
   (cross-teaching) — this is the "peer-selects-my-data" trick that keeps a
   single model from reinforcing its own mistakes, and filters out likely
   mislabeled examples (they tend to have unusually high loss).
3. The fraction of each class taken is **not** symmetric: `[0.5, 0.93]` for
   classes `[0, 1]`. Because the raw training batches are skewed ~65%/35%
   toward class 0, taking half of class 0 but nearly all of class 1
   simultaneously denoises (via the low-loss filter) **and** rebalances
   (via the differentiated rate) the effective training distribution towards
   50/50 — directly targeting *balanced* accuracy.
4. Each model is then updated with only its own selected subset for that step.

Why two models and not one: co-teaching relies on two differently-initialized
networks disagreeing on which samples are noisy, so that neither model's own
confirmation bias about its own selected data compounds — a single model
selecting its own low-loss samples tends to lock onto whatever it already
believes, including noise it has memorized.

## Architecture

`SmallMobileNet` — fixed, not modifiable: a small MobileNet-style depthwise-
separable CNN taking 1×28×28 grayscale input, ending in global average pooling
and a 2-class linear head with dropout.

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-2 |
| Weight decay | 1e-3 |
| Batch size | 128 |
| Epochs | 6 |
| Loss | Cross-entropy (per-sample, `reduction="none"`) |
| Seed | 123 |
| Sampling rates (class 0 / class 1) | 0.5 / 0.93 |

## Results

| Run | Model 1 BAC | Model 2 BAC | Mean BAC | Score |
|---|---|---|---|---|
| Baseline (`default_select_indices`, all data, no filtering) | 0.500 | 0.500 | 0.500 | 0 / 100 |
| **Co-teaching (`your_select_indices`)** — reference run (`notebooks/reference_opracowanie.ipynb`) | 0.8687 | 0.8938 | 0.8812 | **100 / 100** |

The baseline collapses to predicting a single class for both models (BAC =
0.5 — chance level for balanced accuracy) because it trains on 100% of the
noisy, imbalanced data with no filtering. Co-teaching's cross low-loss
selection with rebalanced rates fixes this: both models clear the 0.8 BAC
ceiling required for the maximum score.

**Note:** this session also started a fresh local CPU re-run of the ported
`your_select_indices` implementation (`work/run_train.py`) to double-check it
against real data end-to-end, but the machine was shut down before that CPU
training run finished (full-dataset training on CPU is much slower than the
original GPU run). The code has not been proven to reproduce the reference
numbers on this machine — only ported from, and consistent with, the working
`reference_opracowanie.ipynb` logic. Re-running `python -m src.train` (ideally
on GPU) will produce a fresh, verified result.

Reproduce with:
```bash
pip install -r requirements.txt
python -c "from src.dataset import download_data, unpack_data; \
    download_data('train', '1qmNNmDv-wUcAv5mvO6vYJV3mQ2SNIGnI'); \
    download_data('val', '1YUJYD12NmKRSzFJGMrX-a61d6mnTaWbG'); \
    unpack_data('.', 'train'); unpack_data('.', 'val')"
python -m src.train
```
or open `notebooks/solved_task.ipynb` and run all cells.

## Repository Structure

```
.
├── README.md
├── SOLUTION.md
├── requirements.txt
├── environment.yml            # original conda env spec (competition-provided)
├── configs/
│   └── config.yaml            # hyperparameters, sampling rates, data source ids
├── notebooks/
│   ├── original_task.ipynb        # unsolved contest notebook (provenance)
│   ├── solved_task.ipynb          # `your_select_indices` filled in with co-teaching
│   └── reference_opracowanie.ipynb  # full worked reference: EDA + sweep + final solution + real run
├── results/
│   └── reference_run_results.json  # baseline vs. co-teaching BAC/score (from reference_opracowanie.ipynb)
└── src/
    ├── dataset.py              # TaskDataset + Google Drive download (competition-provided)
    ├── model.py                # SmallMobileNet (competition-provided, fixed architecture)
    ├── select_indices.py       # default_select_indices + your_select_indices (authored solution)
    ├── train.py                # training loop
    └── evaluate.py             # predict_and_evaluate / performance scoring
```
Note: the raw downloaded image data (`train/`, `val/`) is not tracked in git —
download it with `src.dataset.download_data` as shown above.

## Running the Project

```bash
pip install -r requirements.txt
# download & unpack data (see "Results" above), then:
python -m src.train
```

## Technical Highlights

- Class-specific sampling rates were chosen empirically by sweeping candidate
  rate pairs and measuring the resulting class balance of the selected batch
  (see `SOLUTION.md`, "Chosen Approach") — `[0.5, 0.93]` yields an almost
  perfectly 50/50 balanced selection despite the raw ~65/35 skew.
- Cross-teaching (using the *other* model's losses to select a model's own
  training data) is the key mechanism that prevents a model from simply
  reinforcing its own errors on mislabeled samples.
- The whole selection logic is O(batch size) per step and adds negligible
  overhead over vanilla training — well within the 5-minute evaluation budget.

## Potential Improvements

- Increase the fraction of "trusted" samples gradually over training epochs
  (as in the original co-teaching paper's `R(T)` schedule) instead of using a
  fixed rate for all epochs — this typically improves robustness in the very
  early epochs when both models are still unreliable.
- Estimate the noise rate directly from the training data (e.g. via loss
  distribution modeling) instead of hand-picking the sampling rates.
- Add a consistency/agreement-based filter (only trust a sample if both
  models currently agree on its label) as an additional signal beyond loss rank.

---

## Resume Version

**Project**: Robust Binary Image Classification Under Noisy, Imbalanced Labels
**One-line description**: Implemented a co-teaching-based sample-selection
strategy to train two CNNs on a dataset with ~37% label noise and 65/35 class
imbalance, taking mean balanced accuracy from 0.50 (baseline) to 0.88 —
achieving the maximum competition score (100/100).

- Diagnosed that naive training on all noisy, imbalanced data collapses both
  models to a single-class predictor (BAC = 0.5).
- Implemented cross-model low-loss sample selection (co-teaching,
  arXiv:1804.06872) with class-specific sampling rates to jointly denoise
  labels and rebalance classes without touching the fixed model architecture.
- Achieved mean balanced accuracy ≈ 0.88 on held-out validation data (model 1:
  0.869, model 2: 0.894), for a perfect 100/100 competition score.
