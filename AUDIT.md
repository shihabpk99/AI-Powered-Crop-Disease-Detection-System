# Methodological Audit

Independent review of the multi-crop plant disease classification pipeline in this repository.
Every number below was recomputed from the notebooks' saved outputs and from the image files in
`MasterDataset/`, not taken on trust.

**Scope reviewed:** 9 notebooks (`1st_atp` → `Final_attempt`, `Master_attem`,
`Final_Evaluation_on_test`, `real_world_test`), `check_setup.py`, and all 6,551 images in
`MasterDataset/`.

---

## Verdict in one paragraph

The engineering is careful and several things that commonly go wrong here were got right — the
per-backbone normalization is correct in the six base models and in the offline feature path that
produces the headline result, the feature-extraction generators are correctly unshuffled, and the
fusion meta-learner never touches the test set. The serious problem is upstream of the modelling:
the train/val/test split was made per-image rather than per-specimen, over a source set that
contained duplicate files. As a result **26.3% of the test set has a visual near-duplicate in the
training or validation data**, concentrated in exactly the classes that score best. The reported
accuracies — including the headline 96.27% — are therefore optimistic as estimates of field
performance, and the ranking among the three single models is not statistically supported at all.
Separately, one of the four fusion variants applies `1./255` twice (H5) and another is reported from
a model file that no committed cell creates (H6), so the fusion comparison as it currently stands
does not measure what it appears to. None of this requires retraining to fix the reporting; it
requires a clean re-split, a corrected fusion comparison, and one honest re-evaluation.

---

## Findings by severity

### C1 — CRITICAL: near-duplicate images leak across the train/val/test boundary

The split was performed by shuffling the file list inside each class directory and slicing it
(`Master_attem.ipynb`, cell 3). Images were treated as independent samples. They are not: the
source data contains repeated captures of the same physical specimen, plus literal duplicate files.

Measured by perceptual hash (8×8 dHash, Hamming distance against every train and val image):

| dHash distance | Test images with a match in train/val | Share of test set |
|---|---|---|
| = 0 (visually identical) | 56 | 5.65% |
| ≤ 3 (near-identical) | 179 | 18.04% |
| ≤ 5 (clear near-duplicate) | **261** | **26.31%** |
| ≤ 8 (same scene, looser) | 347 | 34.98% |

232 of the 261 matches at distance ≤ 5 are within the same class, which is what makes them harmful:
the model can succeed by recognising a specimen it has already seen rather than a disease.

Independently confirmed by exact MD5 hashing of all 6,551 files:

- 32 test images (3.23%) are **byte-identical** to a train or validation image
- 41 validation images (4.19%) are byte-identical to a training image
- 96 redundant copies exist *inside* the training set alone
- 0 duplicate groups span two different classes, so there is no label-conflict problem

Two distinct causes, both visible in the filenames:

1. **Duplicate files in the source set.** `train/rice_blast/IMG_20231018_143805.jpg` and
   `test/rice_blast/IMG_20231018_143805 - Copy.jpg` are the same bytes. Also `Copy of …`, `…(1)`,
   `…(2)` variants.
2. **Burst captures of one specimen.** 1,619 files (24.7%) carry a `YYYYMMDD_HHMMSS` capture
   timestamp in the filename — 1,128 of them under the camera's `IMG_` prefix, the rest bare.
   Among those, 200 capture-groups (same class, same date, same clock-minute) are split across
   partitions, putting 222 test images (22.4%) in the same capture-minute as a train/val image.
   Example — `potato_bacteria`, 2023-08-15 11:38, ten frames spanning 37 seconds, split 6 train /
   2 val / 2 test. The largest such group is `rice_blast`, 2023-10-18 14:38: 40 frames over 54
   seconds, split 29 train / 6 val / 5 test.

The damage is not evenly spread. Per-class share of the test set with a near-duplicate (distance ≤ 5)
in train/val:

| Class | Affected | Test size | Share |
|---|---|---|---|
| `rice_tungro_virus` | 30 | 37 | **81%** |
| `wheat_blast` | 45 | 60 | **75%** |
| `wheat_leaf_blight` | 32 | 60 | 53% |
| `rice_sheath_blight` | 33 | 75 | 44% |
| `rice_blast` | 47 | 125 | 38% |
| `rice_brown_spot` | 25 | 65 | 38% |
| `wheat_black_point` | 16 | 46 | 35% |
| `rice_healthy` | 8 | 38 | 21% |
| `potato_late_blight` | 4 | 44 | 9% |
| `wheat_healthy` | 3 | 38 | 8% |
| `potato_healthy` | 7 | 98 | 7% |
| `potato_fungi` | 5 | 106 | 5% |
| `wheat_fusarium_foot_rot` | 2 | 38 | 5% |
| `potato_bacteria` | 3 | 86 | 3% |
| `potato_early_blight` | 1 | 76 | 1% |

The rice and wheat classes are the contaminated ones; the potato classes are comparatively clean.
Any per-class F1 discussion in the write-up should account for this, because the classes that look
strongest are the ones with the most leakage.

A full row-level list is in **`audit_leakage_manifest.csv`** (261 rows: test image, nearest
train/val match, dHash distance, whether byte-identical).

**Fix.** Re-split at the specimen level, not the image level. Group by capture session — filename
timestamp bucket for the 24.7% that have one, and perceptual-hash clusters (single-linkage at
distance ≤ 5) for the rest — then assign whole groups to exactly one partition. De-duplicate the
source set first. Then re-evaluate. Expect the headline number to fall; a drop of several points
would be normal and the resulting figure is the one worth publishing.

---

### C1b — note: images are not pre-resized

Worth stating because it is easy to assume otherwise. The 6,551 images are **not** 224×224 on disk.
There are 117 distinct resolutions, ranging from 256×256 up to 4160×3120; the most common are
1500×1500 (1,470 files), 256×256 (1,237), 1440×1080 (1,215) and 1600×1200 (856). Not a single file
is 224×224 — that is the `target_size` of the generators, which resize on load. 39 files are RGBA
rather than RGB, which `load_img(color_mode='rgb')` silently flattens, so it is harmless here.

The reason this matters for C1: the 256×256 subset is a different provenance from the phone-camera
captures, and mixing pre-processed public-dataset images with fresh field captures in one split is
part of how the duplication arose. Any re-split should record the source of each image.

---

### C2 — CRITICAL: the split cannot be regenerated

`Master_attem.ipynb` cell 3 shuffles with no seed, and says so explicitly:

```python
# SHUFFLE: We are going raw here. No seed! Every run is uniquely random.
random.shuffle(images)
```

Re-running that cell produces a different split, silently invalidating every saved model and every
reported metric. Earlier notebooks (`1st_atp`, `2nd_atp`, `3rd_atp`) did seed their splits
(`splitfolders` with `seed=1337` / `seed=42`); the final pipeline lost that.

**Mitigating check — the committed split is the one that was used.** The training logs in
`Final_attempt.ipynb` read `Found 4581 images` / `Found 978 images`, and the directories on disk
contain exactly 4,581 train / 978 val / 992 test. The split was not re-run after training, so the
published numbers are internally consistent. That is luck rather than design.

**Fix.** Seed the split, and additionally commit the split as a manifest (CSV of
`filepath,split`) so it is reproducible independently of the RNG.

---

### H1 — HIGH: the winner was selected on the test set

`Final_Evaluation_on_test.ipynb` evaluates eight configurations on the same 992 test images, and
`Final_attempt.ipynb` reports two further fusion variants against the same test set — ten distinct
configurations in total. The best was reported as the result. Model selection and final reporting
used the same data, so the maximum over ten evaluations is biased upward — an optimistic-max effect
on top of the leakage in C1.

`Final_attempt.ipynb` also evaluates intermediate models against `MasterDataset/test` during
development, so the test set informed iteration.

**Fix.** Choose on validation, report on test once. Every checkpoint already monitors
`val_accuracy`, so the selection machinery is in place — it just needs to be the only thing that
decides, with the test set touched a single time at the end.

---

### H2 — HIGH: the differences between the three single models are not statistically meaningful

With n = 992, the 95% Wilson interval is roughly ±1.6 points. One percentage point is 9.9 images.

| Model | Test acc. | Correct | 95% CI |
|---|---|---|---|
| DenseNet121 (frozen) | 90.93% | 902/992 | 88.98 – 92.56 |
| MobileNetV2 (frozen) | 89.92% | 892/992 | 87.89 – 91.64 |
| EfficientNetB0 (frozen) | 92.44% | 917/992 | 90.63 – 93.93 |
| DenseNet121 (fine-tuned) | 92.54% | 918/992 | 90.74 – 94.02 |
| MobileNetV2 (fine-tuned) | 93.65% | 929/992 | 91.96 – 95.00 |
| EfficientNetB0 (fine-tuned) | 93.15% | 924/992 | 91.40 – 94.56 |
| `Final_Fusion_Boss.keras` (provenance unknown, H6) | 91.23% | 905/992 | 89.31 – 92.83 |
| `The_Three_Fusion`, in-graph features (double-scaled, H5) | 93.45% | 927/992 | 91.73 – 94.83 |
| Decision-level fusion (`The_Ultimate_Fusion_App`) | 94.86% | 941/992 | 93.30 – 96.07 |
| Feature-level fusion (`The_Ultimate_Crop_App`) | 96.27% | 955/992 | 94.90 – 97.28 |

Pairwise (two-proportion z-test):

| Comparison | Difference | p | Conclusion |
|---|---|---|---|
| MobileNetV2 FT vs EfficientNetB0 FT | +0.50 pts | 0.65 | indistinguishable |
| MobileNetV2 FT vs DenseNet121 FT | +1.11 pts | 0.33 | indistinguishable |
| DenseNet121 FT vs DenseNet121 frozen | +1.61 pts | 0.19 | indistinguishable |
| EfficientNetB0 frozen vs DenseNet121 FT | −0.10 pts | 0.93 | indistinguishable |
| Feature fusion vs MobileNetV2 FT | +2.62 pts | 0.008 | significant |
| Feature fusion vs decision-level fusion | +1.41 pts | 0.13 | indistinguishable |
| Decision-level fusion vs MobileNetV2 FT | +1.21 pts | 0.25 | indistinguishable |

So "MobileNetV2 is the best fine-tuned model" rests on **5 images** and does not survive contact
with a confidence interval. Even "fine-tuning beats feature extraction" is not individually
significant per backbone, though the consistent direction across all three is real evidence.

The only comparison that clears significance is feature-level fusion against the best single
model — and C1 applies to it too. Critically, **feature-level fusion does not significantly beat
decision-level fusion** (+1.41 pts, p = 0.13). Any claim that fusing features is superior to fusing
decisions is not supported by this experiment; see H5 and H6 for why the earlier version of that
comparison looked much more convincing than it is.

Two presentation issues compound this. `Final_Evaluation_on_test.ipynb` cell 5 sets
`plt.ylim(80, 95)` and cell 9 sets `plt.ylim(88, 95)`; truncated axes make sub-noise gaps look
decisive. And accuracies are quoted to two decimals (0.01 pt = 0.1 of an image), implying precision
that does not exist.

**Fix.** Report confidence intervals. Use McNemar's test for the paired comparisons — the
predictions are already on the same test images, so it costs nothing. Start bar charts at zero or
show error bars. Quote one decimal.

---

### H3 — HIGH: the fusion meta-learner is trained on in-sample features

`Final_attempt.ipynb` cell 25 extracts features from `MasterDataset/train` using
`DenseNet121_FineTuned`, `MobileNetV2_FineTuned` and the fine-tuned `EfficientNetB0` — all three
models were themselves fine-tuned on those same training images. Cell 26 then trains
`Offline_Boss_V2` on those features.

To be clear about what this is and is not: **it is not test-set leakage.** The extraction
generators use `shuffle=False`, so features and `.classes` labels stay aligned; only train and val
features are saved; the test set is untouched until final evaluation. That part is correct and is
the single most common place this kind of pipeline breaks.

The issue is that this is stacking without out-of-fold predictions. The meta-learner sees
unrealistically clean, over-confident features at training time, because the extractors have
partly memorised those images. At inference it receives noisier features. The result is a
mis-calibrated meta-learner and a gain that is partly an artifact of the mismatch rather than
genuine complementarity between backbones.

**Fix.** Generate the meta-learner's training features out-of-fold: k-fold the training set, fine-tune
the extractors on k−1 folds, extract features for the held-out fold, and train the meta-learner on
the concatenated out-of-fold features. If that is too expensive, hold out a dedicated stacking
split that the base models never see.

---

### H4 — HIGH: `expect_partial()` can silently substitute ImageNet weights

The EfficientNet branch of the fusion model is assembled like this (cells 27, 28, 29 of
`Final_attempt.ipynb`, 6 occurrences overall):

```python
b_eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
...
ext_eff = Model(inputs=eff_inputs, outputs=feat_eff)   # base + GAP only, no Dropout/Dense head
ext_eff.load_weights(os.path.join(MODEL_DIR, "EfficientNetB0_TF_Weights")).expect_partial()
```

Two risks combine. The graph being restored into is a truncated version of the one that was saved
(the `Dropout` and `Dense(15)` layers are gone), which shifts the checkpoint object graph. And
`.expect_partial()` suppresses precisely the diagnostics that would report an incomplete restore.
Because the base was initialised with `weights='imagenet'`, a failed restore does not raise and does
not look wrong — the branch simply contributes generic ImageNet features instead of fine-tuned ones.

The headline 96.27% model depends on this code path. The claim is not necessarily wrong, but it is
currently unverified.

**Fix.** Assert it. Load the weights, run one fixed batch through `ext_eff`, and compare the feature
vector against the same layer of the full `model_eff` built in cell 25 (which loads *without*
`expect_partial()`). If they match, the restore is sound and this finding closes. Drop
`.expect_partial()` and let mismatches raise.

---

### H5 — HIGH: `The_Three_Fusion` scales its input twice

`Final_attempt.ipynb` cell 23 builds an in-graph fusion model that concatenates the three backbones'
pooled features (1024 + 1280 + 1280 = 3584-d) behind a shared input. It routes that input like this:

```python
unified_input = Input(shape=(224,224,3), name="master_input")
# Internal scaler for DenseNet and MobileNet (EfficientNet skips this!)
scaled_input = Rescaling(1./255, name="internal_scaler")(unified_input)

dense_full = load_model(FINAL_DIR/"DenseNet121_FineTuned.keras")
dense_extractor = Model(inputs=dense_full.input, outputs=dense_full.get_layer("global_pooling").output)
feat_dense = dense_extractor(scaled_input)      # <-- 1./255 applied a second time
```

The comment states the intent, and the intent is wrong. `DenseNet121_FineTuned.keras` and
`MobileNetV2_FineTuned.keras` were saved with the rescaler *inside* the graph — their input layer is
`raw_image_input`, immediately followed by `Rescaling(1./255, name="internal_rescaler")`. Feeding
them `scaled_input` therefore divides by 255 twice, so the DenseNet and MobileNet branches see pixels
in [0, 0.0039] instead of [0, 1]. The EfficientNet branch, which correctly skips `internal_scaler`,
is unaffected.

This is the same double-scaling mistake the rest of the notebook takes care to avoid — it simply
lands on the other two backbones instead of EfficientNet, because the rescaler had already been
absorbed into their saved graphs.

Consequence: the 93.45% reported for `The_Three_Fusion` is a measurement of a broken model. Two of
its three branches were fed near-black images, which explains why a three-backbone feature fusion
scored *below* the best of its own components (93.65%). That result should not be cited as evidence
about fusion architecture. The trained meta-layers partly compensated during training — the model
learned around the distortion, since train, val and test all went through the same bad path — which
is exactly why the failure produced a plausible-looking number rather than an obvious collapse.

**Important:** the headline 96.27% model is **not** affected. Cell 29 assembles the production
`The_Ultimate_Crop_App` by feeding `farmer_input` (raw 0–255) straight to each extractor, with no
outer rescaler, which is correct. The offline extraction in cell 25 is likewise correct.

**Fix.** Delete `internal_scaler` from cell 23 and pass `unified_input` directly to
`dense_extractor` and `mobile_extractor`, then re-train and re-report. Better: stop re-deriving
extractors from saved graphs whose preprocessing is implicit, and keep normalization in one
documented place per backbone.

---

### H6 — HIGH: the 91.23% result comes from a model no committed cell creates

`Final_Evaluation_on_test.ipynb` reports **91.23%** for "Master Fusion Boss", loaded from
`D:\Final_Experiment\Final_Fusion_Boss.keras` (source line 846). No notebook in this repository
creates a file of that name. Searching all nine notebooks, the fusion artifacts that *are* written
are `Master_Fusion_Boss_Weights.h5`, `The_Ultimate_Fusion_App`, `The_Three_Fusion_Best_Weights.h5`,
`Offline_Boss_V2.keras` and `The_Ultimate_Crop_App`. `Final_Fusion_Boss.keras` is only ever read.

So there are four fusion models and four numbers, and they need to be kept straight:

| Reported figure | Artifact | Created by | Architecture |
|---|---|---|---|
| 94.86% | `The_Ultimate_Fusion_App` | `Final_attempt` cell 19 | decision-level: three softmax vectors (45-d) → `Dense(64)` → `Dense(15)` |
| 93.45% | `The_Three_Fusion_Best_Weights.h5` | `Final_attempt` cell 23 | feature-level in-graph, 3584-d — **double-scaled, see H5** |
| 96.27% | `The_Ultimate_Crop_App` / `Offline_Boss_V2.keras` | `Final_attempt` cells 25–29 | feature-level offline, per-stream compression to 768-d |
| 91.23% | `Final_Fusion_Boss.keras` | **not in the repository** | unknown |

The consequence is that the fusion comparison in the evaluation notebook silently substitutes the
unprovenanced 91.23% model for the 94.86% decision-level model that the pipeline actually produced.
Reported that way, feature fusion appears to beat decision fusion by 5.0 points (p < 0.001). Against
the decision-level model that the code does create, the gap is 1.4 points and not significant
(p = 0.13). This single bookkeeping error is the difference between a headline claim and no claim.

**Fix.** Either locate the cell that produced `Final_Fusion_Boss.keras` and commit it, or drop the
91.23% row entirely and report the decision-level baseline as 94.86% from
`The_Ultimate_Fusion_App`. Then restate the fusion comparison against that number. Going forward,
name each artifact after the cell that writes it and never evaluate a file that no committed code
produces.

---

### M1 — MEDIUM: ambiguous EfficientNet weight provenance

Four EfficientNet artifacts are referenced: `EfficientNetB0_Base_Weights.h5`,
`EfficientNetB0_FineTuned_Weights.h5`, `EfficientNetB0_Final_Weights.h5`, and
`EfficientNetB0_TF_Weights`. `Final_Evaluation_on_test.ipynb` reports the 93.15% fine-tuned figure
from `EfficientNetB0_Final_Weights.h5`, while the fusion model consumes `EfficientNetB0_TF_Weights`.
Whether these hold identical weights is not established anywhere. The paper trail matters, because
the fusion result and the single-model result would then describe different EfficientNets.

### M2 — MEDIUM: brittle weight loading dependent on replicating trainable flags

Restoring the `.h5` weights requires rebuilding the architecture *and* reproducing the exact
`trainable` flags, as the notebook itself documents:

```python
# 🚨 THE FIX: We MUST replicate the exact trainable state from Phase 2
base_efficient.trainable = True
for layer in base_efficient.layers[:-50]:
    layer.trainable = False
```

An earlier attempt in the same notebook fails with `ValueError: axes don't match array` — the saved
traceback is still in `Final_Evaluation_on_test.ipynb` cell 3. Any future reader who rebuilds with
different flags gets silently or loudly wrong weights. DenseNet and MobileNet avoid this entirely by
saving full `.keras` models; EfficientNet should too.

### M3 — MEDIUM: class weighting applied inconsistently

The base models pass `class_weight=class_weight_dict` (computed `balanced` over the training
labels). The fusion meta-learner in cell 26 does not. Class support ranges from 239
(`rice_tungro_virus`) to 828 (`rice_blast`), a 3.46:1 ratio, so the two stages optimise different
objectives.

### M4 — MEDIUM: single run per configuration

Every configuration was trained once. `Final_attempt.ipynb` locks `SEED = 42` for Python, NumPy and
TensorFlow, which is good for reproducibility but yields no variance estimate. Run-to-run spread
from GPU non-determinism and augmentation ordering is plausibly ±1 point on 992 images — the same
size as the differences being interpreted in H2. Three seeds per configuration, reported as
mean ± sd, would settle it.

### M5 — MEDIUM: nothing in the repository runs after cloning

- Every path is an absolute Windows path. Counting literal occurrences across all nine notebooks'
  cell sources, whether the prefix stands alone or is followed by a sub-path: `D:\00_Thesis_Split`
  101, `D:\MasterDataset` 57, `D:\MasterModels` 45, `D:\Final_Experiment` 42, plus `D:\00` (the raw
  source pool) 157. The dataset in this repository actually lives at `MasterDataset/`, so no
  notebook resolves as committed.
- No trained model artifacts are committed — no `.keras`, `.h5`, or `saved_model.pb` anywhere.
  `D:\Final_Experiment` is where all ten reported results were written and it is outside the
  repository, so none of them can be reproduced or even spot-checked by a reader. This is also what
  makes H4 and H6 unresolvable from the repository alone.
- No `requirements.txt` or `environment.yml`, and no pinned TensorFlow version. The saved tracebacks
  point at `miniconda3/envs/tf_gpu` on Keras 2.x; the code will not run unmodified on Keras 3.
- No `LICENSE`.

### L1 — LOW: repository hygiene

`.git` is 2.1 GB with the full image dataset committed and no `.gitignore`. That exceeds GitHub's
recommended repository size and will make cloning slow. Consider Git LFS, or host the dataset
externally (Zenodo, Kaggle) and commit only the split manifest.

### L2 — LOW: `real_world_test.ipynb` is a stale artifact

It loads `DenseNet121_Fresh_Phase2_FineTuned.h5` from `D:\00_Thesis_Final_Models` with a different
head (`BatchNormalization` → `Dense(256)` → `Dropout(0.4)`) than the final pipeline's
(`Dropout(0.3)` → `Dense(15)`), and reads from `D:\Real_World_Rice`, which is not in the repository.
It prints diagnoses with no ground truth, so it demonstrates inference but measures nothing. Either
promote it to a real held-out field evaluation with labels, or mark it clearly as a demo.

---

## What the audit confirms as correct

Worth recording, both because it is good work and so a reader does not re-litigate it:

- **Per-backbone normalization is right in the paths that matter, which is the subtle one.**
  DenseNet121 and MobileNetV2 get an internal `Rescaling(1./255)` layer; EfficientNetB0 receives raw
  0–255 pixels because Keras EfficientNet normalizes inside the graph. Double-scaling EfficientNet is
  the standard mistake here and it was avoided in all six base models, in the offline feature
  extraction of cell 25, and in the production `The_Ultimate_Crop_App` of cell 29 — which is the path
  the headline 96.27% depends on. The one place it goes wrong is cell 23's `The_Three_Fusion`, where
  the *other two* backbones get scaled twice; that is H5 and it does not touch the headline result.
- **Evaluating with `ImageDataGenerator()` and no rescale is therefore correct**, not a bug, because
  the scaling lives inside the saved models.
- **Feature/label alignment is correct.** The extraction generators use `shuffle=False`, so
  `predict()` output order matches `.classes`. With `shuffle=True` this would have produced
  scrambled labels and a plausible-looking but meaningless meta-learner.
- **The test set is genuinely excluded from meta-learner training** — only `train_feat_*` and
  `val_feat_*` are saved and loaded.
- **`shuffle=False` and `test_gen.reset()` before every `predict()`**, so confusion matrices and
  classification reports align with true labels.
- **Two-phase transfer learning is textbook**: freeze the backbone and train the head at lr 1e-3,
  then unfreeze the last 50 layers at lr 1e-5 with `EarlyStopping`, `ReduceLROnPlateau` and
  `ModelCheckpoint(save_best_only=True, monitor='val_accuracy')`. The warning against using 1e-3
  during fine-tuning is correct.
- **Batch normalization is held in inference mode during fine-tuning** via
  `base(inputs, training=False)`. This is deliberate and matches the Keras transfer-learning
  guidance; it is a design choice, not an oversight.
- **Class imbalance is addressed** for the base models with `compute_class_weight('balanced')`.
- **Split proportions are as intended**: 69.9% / 14.9% / 15.1%.

---

## Recommended order of work

1. **Fix the fusion bookkeeping (H6)** — half an hour, and it changes a headline claim. Replace the
   91.23% row with the 94.86% decision-level result that the code actually produces, then restate the
   feature-vs-decision comparison honestly.
2. **Verify H4** — one fixed batch through both EfficientNet graphs. Cheap, and everything
   downstream of the fusion claim depends on it.
3. **Remove the double rescaler in cell 23 and re-run it (H5)**, so the third fusion variant measures
   an architecture rather than a preprocessing bug.
4. **De-duplicate and re-split at specimen level (C1, C2)**, seeded, with a committed manifest that
   also records each image's source resolution and origin (C1b).
5. **Re-evaluate every configuration once** on the clean test set. Report accuracy with confidence
   intervals and McNemar for paired comparisons (H2).
6. **Retrain the meta-learner out-of-fold (H3)**, with `class_weight` for consistency (M3).
7. **Three seeds per configuration**, report mean ± sd (M4).
8. **Make the repository runnable (M5)**: relative paths, `requirements.txt` with pinned
   TensorFlow, committed or externally hosted weights, `LICENSE`, `.gitignore`.

Steps 1, 2 and 5 alone convert the current results from "unverifiable" to "defensible". The expected
outcome is a lower headline accuracy that is actually supportable — a stronger position than 96.27%
that does not survive questioning about the test set.

---

## How to reproduce this audit

```bash
# exact duplicates
find MasterDataset -type f \( -iname '*.jpg' -o -iname '*.png' \) -print0 \
  | xargs -0 md5sum > hashes.txt
# then group by hash and flag any group spanning two split directories

# near duplicates: 8x8 dHash per image, Hamming distance from each test image
# to every train/val image; flag distance <= 5

# capture-session leakage: parse YYYYMMDD_HHMMSS from filenames, group by
# (class, date, HH:MM), flag any group spanning more than one split

# resolutions: PIL Image.open(...).size over all files, counted
```

Row-level output: `audit_leakage_manifest.csv`.
