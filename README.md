# Multi-Crop Plant Disease Classification

Transfer-learning and model-fusion experiments for classifying 15 disease and healthy conditions
across three staple crops — potato, rice, and wheat — from single leaf images.

Three ImageNet backbones (DenseNet121, MobileNetV2, EfficientNetB0) are trained in two phases, then
combined by fusion at both the feature level and the decision level. The best configuration reaches
**96.3% accuracy on a held-out test set of 992 images** — with important caveats about that number
documented in [Known limitations](#known-limitations) and in [`AUDIT.md`](AUDIT.md).

> **Read this first.** An independent methodological audit of this repository found that 26% of the
> test set has a visual near-duplicate in the training data, because the split was made per-image
> over a source set containing repeat captures of the same specimens. The accuracies below are
> therefore optimistic as estimates of real-world field performance; the differences between the
> three single models are within statistical noise; and feature-level fusion does not significantly
> outperform decision-level fusion once it is compared against the right model. Full analysis and
> remediation plan: [`AUDIT.md`](AUDIT.md). This section is kept prominent deliberately — the numbers
> are reported as measured, with their limits stated.

---

## Dataset

6,551 images across 15 classes, split 70 / 15 / 15 at the image level.

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| `potato_bacteria` | 398 | 85 | 86 | 569 |
| `potato_early_blight` | 351 | 75 | 76 | 502 |
| `potato_fungi` | 489 | 105 | 106 | 700 |
| `potato_healthy` | 452 | 97 | 98 | 647 |
| `potato_late_blight` | 202 | 43 | 44 | 289 |
| `rice_blast` | 579 | 124 | 125 | 828 |
| `rice_brown_spot` | 296 | 63 | 65 | 424 |
| `rice_healthy` | 175 | 37 | 38 | 250 |
| `rice_sheath_blight` | 350 | 75 | 75 | 500 |
| `rice_tungro_virus` | 167 | 35 | 37 | 239 |
| `wheat_black_point` | 212 | 45 | 46 | 303 |
| `wheat_blast` | 280 | 60 | 60 | 400 |
| `wheat_fusarium_foot_rot` | 175 | 37 | 38 | 250 |
| `wheat_healthy` | 175 | 37 | 38 | 250 |
| `wheat_leaf_blight` | 280 | 60 | 60 | 400 |
| **Total** | **4,581** | **978** | **992** | **6,551** |

Potato accounts for 2,707 images, rice 2,241, wheat 1,603. Class support is imbalanced at 3.46:1
between the largest (`rice_blast`, 828) and smallest (`rice_tungro_virus`, 239), addressed during
training with `compute_class_weight('balanced')`.

Images are JPEG at their original capture resolutions — they are **not** pre-resized. There are 117
distinct resolutions from 256×256 up to 4160×3120, most commonly 1500×1500 (1,470 files), 256×256
(1,237), 1440×1080 (1,215) and 1600×1200 (856). Resizing to 224×224 happens on load, via the
generators' `target_size`. 39 files are RGBA and are flattened to RGB by `load_img`.

Layout:

```
MasterDataset/
├── train/   <15 class directories>
├── val/     <15 class directories>
└── test/    <15 class directories>
```

---

## Method

### Preprocessing

Normalization differs per backbone, which matters:

- **DenseNet121, MobileNetV2** — a `Rescaling(1./255)` layer is built *into* the model graph, so
  saved models accept raw 0–255 pixels.
- **EfficientNetB0** — receives raw 0–255 pixels directly, because Keras `EfficientNetB0` performs
  its own normalization internally. Applying `1./255` here would double-scale the input.

Because scaling lives inside the models, evaluation uses a bare `ImageDataGenerator()` with no
`rescale` argument. This is intentional.

One caveat: because the rescaler is baked into the DenseNet and MobileNet graphs, any code that
re-derives a feature extractor from those saved models must pass **raw** pixels. `The_Three_Fusion`
gets this wrong (audit H5); the production model does not.

Training augmentation: rotation ±25°, width/height shift 0.15, zoom 0.15, shear 0.1, brightness
0.8–1.2, horizontal flip, `fill_mode='nearest'`. Validation and test data are never augmented.

### Two-phase transfer learning

**Phase 1 — feature extraction.** Backbone frozen, custom head trained
(`GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(15, softmax)`), Adam at lr 1e-3, batch size 16,
up to 25 epochs.

**Phase 2 — fine-tuning.** Last 50 backbone layers unfrozen, Adam at lr 1e-5. The low learning rate
is essential; 1e-3 here causes catastrophic forgetting. Batch normalization is deliberately held in
inference mode via `base(inputs, training=False)`, per the Keras transfer-learning guidance.

Both phases use `ModelCheckpoint(monitor='val_accuracy', save_best_only=True)`,
`EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and
`ReduceLROnPlateau(factor=0.5)`.

### Fusion strategies

**Feature-level fusion** (the best performer). Penultimate `GlobalAveragePooling2D` features are
extracted offline from all three fine-tuned backbones — 1024-d from DenseNet121, 1280-d from
MobileNetV2, 1280-d from EfficientNetB0. A meta-classifier compresses each stream through
`Dense(256)` → `BatchNormalization` → `ReLU` → `Dropout(0.3)`, concatenates the three (768-d), then
`Dense(128, relu)` → `Dropout(0.2)` → `Dense(15, softmax)`. Trained on training-set features with
validation-set features for early stopping; the test set is not involved.

Extraction generators use `shuffle=False` so that features stay aligned with
`generator.classes` labels.

**Decision-level fusion.** The three fine-tuned models keep their softmax heads; their outputs are
concatenated (45-d) and passed to `Dense(64, relu)` → `Dropout(0.2)` → `Dense(15, softmax)`, trained
end-to-end with the backbones frozen. This reaches 94.9%, which is not significantly below
feature-level fusion.

**A third variant, `The_Three_Fusion`,** concatenates the raw 3584-d pooled features inside a single
graph. It reaches 93.5% — but it contains a preprocessing bug (it applies `1./255` twice to two of
its three branches), so that number measures a broken model rather than an architecture. See H5 in
[`AUDIT.md`](AUDIT.md). It is listed below for completeness only.

The final artifact stitches the three feature extractors and the meta-classifier into one
`tf.saved_model` graph accepting a raw 224×224×3 image.

---

## Results

Single evaluation on the 992-image held-out test set. Confidence intervals are 95% Wilson; at this
sample size one percentage point is 9.9 images.

| Model | Configuration | Test accuracy | 95% CI |
|---|---|---|---|
| DenseNet121 | frozen backbone | 90.9% | 89.0 – 92.6 |
| MobileNetV2 | frozen backbone | 89.9% | 87.9 – 91.6 |
| EfficientNetB0 | frozen backbone | 92.4% | 90.6 – 93.9 |
| DenseNet121 | fine-tuned | 92.5% | 90.7 – 94.0 |
| MobileNetV2 | fine-tuned | 93.7% | 92.0 – 95.0 |
| EfficientNetB0 | fine-tuned | 93.2% | 91.4 – 94.6 |
| `The_Three_Fusion` | 3 models, in-graph features — *has a preprocessing bug* | 93.5% | 91.7 – 94.8 |
| Decision-level fusion | 3 models, softmax | 94.9% | 93.3 – 96.1 |
| **Feature-level fusion** | **3 models, offline features** | **96.3%** | **94.9 – 97.3** |

Reading these responsibly:

- **The three fine-tuned models are statistically indistinguishable.** MobileNetV2's 0.5-point lead
  over EfficientNetB0 is 5 images (p = 0.65); its 1.1-point lead over DenseNet121 is p = 0.33. This
  repository does not establish a best single backbone.
- **Fine-tuning helps, consistently but modestly.** Each backbone improves by 0.7–3.7 points. No
  single improvement is individually significant, but the direction holds for all three, which is
  real evidence.
- **Feature-level fusion beats the best single model** by 2.6 points (p = 0.008) — the only
  comparison here that clears significance.
- **But it does not beat decision-level fusion.** The gap is 1.4 points (p = 0.13). This repository
  therefore does **not** support the claim that fusing features is better than fusing decisions. An
  earlier version of the evaluation appeared to show a decisive 5-point gap; that came from comparing
  against a different model file whose training code is not in the repository. See H6 in
  [`AUDIT.md`](AUDIT.md).
- **All of these figures are inflated to an unquantified degree** by the near-duplicate leakage
  described below.

---

## Known limitations

Established by the audit in [`AUDIT.md`](AUDIT.md), which recomputed every figure from the notebook
outputs and all 6,551 image files.

**Near-duplicate leakage across splits.** The split shuffles filenames within each class and slices,
treating images as independent. They are not: 24.7% of files carry `YYYYMMDD_HHMMSS` capture
timestamps showing burst sequences of the same specimen seconds apart, and the source set contains
literal duplicate files (`- Copy`, `(1)`, `Copy of …`). Measured against train and validation data:

| Overlap measure | Test images affected | Share of test set |
|---|---|---|
| Byte-identical (MD5) | 32 | 3.2% |
| Visually identical (dHash = 0) | 56 | 5.7% |
| Near-duplicate (dHash ≤ 5) | 261 | 26.3% |

Leakage is heavily class-dependent — `rice_tungro_virus` 81%, `wheat_blast` 75%,
`wheat_leaf_blight` 53%, while `potato_early_blight` is 1% and `potato_bacteria` 3%. Per-class
metrics should be read with this in mind, since the strongest-looking classes are the most
contaminated. Row-level detail: [`audit_leakage_manifest.csv`](audit_leakage_manifest.csv).

**The split is not regenerable.** It was produced with `random.shuffle()` and no seed. The committed
directories are verified to be the exact split used for training (image counts match the training
logs: 4,581 / 978 / 992), so published numbers are internally consistent — but the split cannot be
reconstructed from code.

**Model selection used the test set.** Ten configurations were evaluated on the same test images and
the best reported, so the headline figure carries an optimistic-max bias on top of the leakage.

**One fusion variant has a preprocessing bug.** `The_Three_Fusion` (93.5%) applies `Rescaling(1./255)`
to inputs that are then fed to saved models which already rescale internally, so two of its three
branches receive pixels in [0, 0.0039]. Its accuracy is not evidence about fusion. The headline 96.3%
model is a different code path and is not affected. Audit finding H5.

**One reported number has no code behind it.** `Final_Evaluation_on_test.ipynb` evaluates
`Final_Fusion_Boss.keras` at 91.2%, and no committed notebook creates that file. It has been excluded
from the results table above. Audit finding H6.

**The fusion meta-learner is trained on in-sample features.** Features come from the same training
images the extractors were fine-tuned on. This is *not* test-set leakage — only train and validation
features are used — but it is stacking without out-of-fold predictions, so the meta-learner trains on
over-confident features and is likely mis-calibrated.

**Single run per configuration.** Seeds are locked at 42, so results are reproducible, but there is
no variance estimate. Run-to-run spread is plausibly the same size as the differences being compared.

**One EfficientNet weight-restore path is unverified.** The fusion model loads EfficientNet weights
into a truncated graph with `.expect_partial()`, which suppresses incomplete-restore diagnostics.
Because the backbone is initialised with `weights='imagenet'`, a silent failure would leave generic
features in place without raising. See finding H4 in [`AUDIT.md`](AUDIT.md) for a one-batch check
that settles it.

**Reproducibility gaps.** Notebooks use absolute Windows paths (`D:\MasterDataset`,
`D:\Final_Experiment`) that do not match this repository's layout; no trained weights are committed;
there is no pinned dependency list. See below.

---

## Repository contents

| Path | Description |
|---|---|
| `Final_attempt.ipynb` | **Main pipeline.** Seeded run: both phases for all three backbones, offline feature extraction, all three fusion variants, final export. |
| `Final_Evaluation_on_test.ipynb` | Test-set evaluation — accuracy, classification reports, confusion matrices. Note H6: one of the models it loads is not built by any committed cell. |
| `Master_attem.ipynb` | Dataset assembly and the train/val/test split (see C2 in the audit). |
| `1st_atp.ipynb` – `5th_atp_hybrid_mas.ipynb` | Earlier iterations, kept for provenance. Superseded by `Final_attempt.ipynb`. |
| `real_world_test.ipynb` | Unlabelled inference demo on field photos. Loads an older checkpoint with a different head; demonstrative only. |
| `check_setup.py` | Verifies TensorFlow, GPU visibility, and helper libraries. |
| `MasterDataset/` | The image dataset, pre-split. |
| `AUDIT.md` | Full methodological audit. |
| `audit_leakage_manifest.csv` | The 261 test images with a near-duplicate in train/val, with distances. |
| `*.png` | Confusion matrices, training curves, comparison charts. |

Note that the earlier `Nth_atp` notebooks reference paths (`D:\00_Thesis_Split`, `D:\MasterModels`)
and models from superseded experiments; their reported numbers do not correspond to the results
table above.

---

## Setup

```bash
python check_setup.py
```

Confirms TensorFlow, GPU availability, and that Matplotlib, NumPy, OpenCV and scikit-learn are
present.

The notebooks were developed against **TensorFlow 2.x with Keras 2** (a conda environment named
`tf_gpu`). They use `tensorflow.keras.preprocessing.image.ImageDataGenerator`, which is removed in
Keras 3 — they will not run unmodified on a Keras 3 install. A GPU is strongly recommended; epochs
run about 82–265 s on 4,581 images at batch size 16 depending on backbone and phase.

Suggested environment:

```bash
conda create -n tf_gpu python=3.9
conda activate tf_gpu
pip install "tensorflow<2.16" matplotlib numpy opencv-python scikit-learn seaborn pandas split-folders
```

`tensorflow<2.16` pins Keras 2.

### Before running

Two changes are needed, since the notebooks are not currently portable:

1. **Fix the paths.** Replace `D:\MasterDataset` with `MasterDataset` (or your own absolute path) and
   point `D:\Final_Experiment` at a writable output directory. Across the notebooks `D:\MasterDataset`
   appears 57 times and `D:\Final_Experiment` 42 times; the superseded notebooks add
   `D:\00_Thesis_Split` (101) and `D:\MasterModels` (45).
2. **Expect to retrain.** No model weights are committed, so `Final_Evaluation_on_test.ipynb` cannot
   run until `Final_attempt.ipynb` has produced the checkpoints.

Order of execution: `Final_attempt.ipynb` end to end, then `Final_Evaluation_on_test.ipynb`.

---

## If you are building on this

The audit's recommended sequence, in priority order:

1. Settle the fusion bookkeeping (audit H6) — report the decision-level baseline from the model the
   code actually produces, and restate the feature-vs-decision comparison against it.
2. Verify the EfficientNet weight restore in the fusion path (audit H4) — cheap, and the headline
   result depends on it.
3. Remove the double rescaler from `The_Three_Fusion` and re-run it (audit H5) so that variant
   measures an architecture instead of a bug.
4. De-duplicate the source images, then re-split **by specimen** rather than by image: group files by
   capture session (filename timestamp bucket, plus perceptual-hash clusters at distance ≤ 5) and
   assign whole groups to one partition. Seed it and commit a `filepath,split` manifest.
5. Re-evaluate once on the clean test set. Report confidence intervals, and use McNemar's test for
   paired comparisons — the predictions are already on shared test images.
6. Retrain the fusion meta-learner on out-of-fold features, with `class_weight` applied for
   consistency with the base models.
7. Run three seeds per configuration and report mean ± sd.

Expect the headline accuracy to drop. A lower, clean number is a stronger result than 96.3% that
cannot withstand a question about the test set.

---

## License

No license file is currently present. Add one before sharing or reusing this work.
