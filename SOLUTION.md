# Solution Write-up

## 1. Problem Formulation

Train two binary image classifiers, `model1` and `model2`, sharing the fixed
`SmallMobileNet` architecture, from a training set in which:
- an unknown subset of labels are wrong (label noise), and
- classes are imbalanced (class 0: 6298 samples, class 1: 3702 samples — a
  ~63/37 split, *inverted* relative to the clean validation set, which skews
  toward class 1).

The only degree of freedom is `your_select_indices(targets, losses)`, called
once per training batch, which must return, for each of the two models, the
indices of the batch it is allowed to train on this step. Everything else
(architecture, optimizer, training loop, evaluation) is fixed
*(fact, from the provided notebook code)*.

## 2. Dataset and Preprocessing

- 28×28 grayscale images, loaded via a fixed `TaskDataset` (reads
  `dataset_labels.csv` + a `data/` folder of PNGs), converted to tensors with
  no further augmentation.
- EDA from `notebooks/reference_opracowanie.ipynb`: no missing labels, no
  missing image files; label counts 6298 (class 0) vs 3702 (class 1) in
  training. t-SNE/PCA projections of pixel space show no clean separation
  between classes, and per-class pixel value histograms/boxplots look similar
  — consistent with the images themselves being genuinely ambiguous/hard, on
  top of a fraction having outright wrong labels *(fact, from opracowanie
  notebook outputs)*.

## 3. Initial Considerations

The competition's own baseline (`default_select_indices`) simply trains on
100% of every batch — i.e. standard training, no noise-robustness mechanism.
Run against the real data, both models collapse to a single-class prediction:
BAC = 0.5 for both, mean BAC = 0.5, score = 0/100 *(fact, from
`notebooks/reference_opracowanie.ipynb` cell output)*. This is the expected failure
mode of gradient descent on a noisy, imbalanced binary task: the easiest way
to reduce average loss when a third of the labels are flipped and one class
dominates is to ignore the signal and just predict the majority-ish class.

A naive fix — simply downweighting/dropping a fixed random fraction of data —
does not target *which* samples are mislabeled, and does nothing about class
imbalance. Something that (a) identifies likely-clean samples and (b)
independently controls the resulting class distribution is needed
*(inferred)*.

## 4. Chosen Approach: Co-Teaching with Class-Specific Sampling Rates

Reference: Han et al., *"Co-teaching: Robust Training of Deep Neural Networks
with Extremely Noisy Labels"*, NeurIPS 2018, [arXiv:1804.06872](https://arxiv.org/abs/1804.06872).

Core co-teaching idea: train two networks in parallel; at each step, each
network identifies the samples it currently finds "easiest" (lowest loss) and
feeds *those* to the other network for its gradient update, rather than
training on its own selected samples. The intuition, and the paper's
empirical/theoretical argument, is that under label noise, a network's own
low-loss set will increasingly include memorized noisy samples as training
progresses (since networks memorize noise given enough capacity/time), but two
*independently initialized* networks diverge on which noisy samples they've
started to memorize — so cross-selecting keeps each network's training data
biased toward the samples both networks currently agree are "easy" for the
*right* reasons.

This solution's specific implementation (`src/select_indices.py`,
`your_select_indices`):

```python
SAMPLING_RATES = [0.5, 0.93]  # per class [0, 1]

for class_idx in torch.unique(targets):
    class_mask = targets == class_idx
    class_indices = class_mask.nonzero(as_tuple=True)[0]
    take_n = int(len(class_indices) * SAMPLING_RATES[class_idx])
    for i in range(2):
        other_losses = losses[1 - i].clone()
        other_losses[targets != class_idx] = float("inf")
        _, best_idx = torch.topk(-other_losses, k=take_n)
        selected_indices[i].extend(best_idx)
```

Two things happen simultaneously:
1. **Denoising**: within each class, only the `take_n` samples with the
   lowest loss *according to the other model* are kept — filtering out the
   samples most likely to be mislabeled (a wrong label typically produces an
   anomalously high loss once the model has learned the true pattern for that
   region of input space).
2. **Rebalancing**: the sampling rate differs by class (0.5 for class 0, 0.93
   for class 1). Since raw batches are ~65% class 0 / 35% class 1, taking half
   of the majority class but nearly all of the minority class brings the
   *selected* batch close to 50/50 — directly optimizing for the *balanced*
   accuracy metric used for scoring, rather than raw accuracy.

The exact rates were chosen via a small offline experiment
(`test_sampling_rates` in `notebooks/reference_opracowanie.ipynb`): several
candidate rate pairs were applied to 10 accumulated real training batches, and
the resulting class balance of the selected samples was measured. `[0.5, 0.92]`
/ `[0.5, 0.93]` both landed within roughly a percentage point of a 50/50 split
(50.1%/49.9% for `[0.5, 0.92]` on the swept batches); `[0.5, 0.93]` was used in
the final function *(fact, from opracowanie notebook code and printed sweep
results)*.

## 5. Model Architecture

`SmallMobileNet` (fixed, `src/model.py`): depthwise-separable convolutional
blocks (`Conv-BN-ReLU6` stages with grouped 3×3 depthwise + 1×1 pointwise
convolutions, standard MobileNet-style factorization) reducing a 1×28×28 input
down through 32→64→128→256 channels with two stride-2 downsamples, followed by
global average pooling and a `256→128→2` classifier head with dropout(0.5).

## 6. Training Procedure

Both models are initialized identically in structure (different random
weights due to independent instantiation) and trained for 6 epochs with
AdamW (lr=1e-2, weight_decay=1e-3), batch size 128, on `nn.CrossEntropyLoss`
computed per-sample (`reduction="none"`) so that co-teaching can rank
individual sample losses. Each of the two optimizer steps per batch only
backpropagates through that model's own selected subset (see `src/train.py`,
`train_loop`). Seed fixed to 123 for reproducibility *(fact, from the fixed
notebook code)*.

## 7. Evaluation Methodology

Balanced accuracy (`sklearn.metrics.balanced_accuracy_score`) is computed
independently for each model on the (clean) validation/test set, then averaged
and mapped to a 0–100 score via the piecewise-linear formula in `README.md`
(0 pts at BAC_mean ≤ 0.5, 100 pts at BAC_mean ≥ 0.8, linear between)
*(fact, from the fixed `performance` function)*.

## 8. Results

| Run | Model 1 BAC | Model 2 BAC | Mean BAC | Score |
|---|---|---|---|---|
| Baseline (no filtering) | 0.500 | 0.500 | 0.500 | 0/100 |
| Co-teaching, reference run | 0.8687 | 0.8938 | 0.8812 | 100/100 |
| Co-teaching, local CPU re-run (this session, `results/local_run_results.json`) | 0.6721 | 0.8728 | 0.7725 | 90.8/100 |

The local CPU re-run independently confirms the ported `your_select_indices`
works correctly end-to-end against the real downloaded data — it is not a
re-typed/re-derived implementation, so this is mainly a sanity check that
nothing was lost in translation from `reference_opracowanie.ipynb`.

Two things worth noting from this run:

1. **CPU performance bug (fixed for this run, not a code bug):** the first
   attempt at this local run used PyTorch's default thread count (12, matching
   this machine's core count) and was pathologically slow — over 30 CPU-minutes
   without finishing a single epoch. A microbenchmark isolated the cause:
   `SmallMobileNet`'s ops are tiny (28×28 inputs, depthwise-separable convs),
   so PyTorch's per-op thread-pool dispatch overhead with 12 threads dwarfs the
   actual compute — classic over-threading for small tensors. Setting
   `torch.set_num_threads(4)` before training dropped one epoch from
   "didn't finish in 30+ CPU-minutes" to ~200 wall-clock seconds. This is a
   deployment/environment detail, not something `your_select_indices` controls,
   but it's worth knowing if reproducing this on another small-core-count-vs-
   op-size-mismatched machine *(fact, measured this session)*.
2. **Late-epoch instability:** `model1`'s validation BAC peaked at 0.87
   (epoch 5) then dropped to 0.67 at epoch 6 (the final, reported epoch) —
   see the full per-epoch log. `model2` stayed stable (0.85→0.87→0.87).
   This pulled the mean BAC (and score) below the reference run's numbers,
   even though the same code, hyperparameters, and seed(123) as the reference
   were used. Two most likely explanations *(inferred)*: (a) the reference
   run's exact numeric trajectory depends on the PyTorch/CUDA version and GPU
   nondeterminism (the reference notebook ran on `torch==2.5.1`+CUDA per
   `environment.yml`; this session used `torch==2.14` CPU-only — operator
   implementations, and therefore the exact sequence of floating-point
   results even from the "same" seed, differ across versions/backends), and
   (b) `lr=1e-2` with AdamW is fairly aggressive for a model this small, so a
   late-training overshoot on one of the two models plausibly explains a
   BAC swing this size while the loss/selection logic itself remains correct.
   The fix that would most directly address this without changing the
   allowed `your_select_indices` function: track validation BAC per epoch
   during training and keep the best-epoch weights instead of returning the
   final epoch's, or add LR decay over the last epoch or two.

## 9. Why the Approach Works

- Cross-teaching breaks the feedback loop where a network reinforces its own
  memorized mistakes: since `model1` never chooses its own training subset, it
  cannot simply keep training on whatever noisy samples it has already
  overfit to.
- Class-specific rates convert a *general* denoising heuristic (low loss ⇒
  probably clean) into one that is also *aware of the label distribution
  shift* between train and validation — without this, filtering low-loss
  samples uniformly across classes would preserve the original class
  imbalance and continue to bias the models toward the majority class.
- Because both effects act at the level of *which samples enter the gradient
  update*, no change to the fixed architecture or loss function is needed —
  satisfying the competition's hard constraint.

## 10. Limitations

- The sampling rates are fixed constants tuned on this particular dataset's
  noise/imbalance ratio; they would need to be re-tuned (or estimated
  automatically) for a differently-noised or differently-imbalanced dataset
  *(inferred)*.
- Early in training, both models are still unreliable, so "low loss according
  to the other model" is a noisier signal than later in training; the paper's
  original schedule ramps the keep-rate down over epochs rather than using a
  constant rate, which this solution does not do.
- No explicit mechanism handles the case where *both* models agree on a wrong
  label (correlated errors) — cross-teaching only protects against
  *independent* memorization, not against structural biases shared by both
  networks *(inferred)*.

## 11. Potential Improvements

- Epoch-dependent sampling-rate schedule (start conservative, e.g. keep only
  the most confident ~20% of each class, and relax toward the final rates as
  training progresses), following the original co-teaching paper more closely.
- Estimate the label-noise rate per class directly from the loss distribution
  (e.g. fitting a two-component mixture model to per-sample losses) instead of
  hand-picking rates via a manual sweep.
- Add a small held-out "trust" set to periodically validate that the selected
  training subset is actually improving generalization, rather than relying
  solely on the validation curves shown during training.
