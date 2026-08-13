# Dataset Sources and Provenance

This project does not claim the leaf images as a newly collected original dataset. The repository's `MasterDataset/` is an **integrated research dataset** created by selecting, renaming, reorganizing, and splitting images from four publicly available sources.

The original download archives were used to confirm the source titles and directory contents. Before redistributing or reusing any image, consult the current terms and licence on its original source page.

## Original sources

| Source | Platform | How it was used | Official page |
|---|---|---|---|
| Rice Leaf Diseases Dataset | Kaggle | Natural-image rice classes, including blast, brown spot, healthy leaf, and sheath blight | [Kaggle dataset](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) |
| Paddy Disease Classification / Paddy Doctor | Kaggle | A selected subset was used to supplement rice categories, including tungro-related images | [Kaggle competition](https://www.kaggle.com/competitions/paddy-disease-classification) |
| Potato Leaf Disease Dataset in Uncontrolled Environment | Mendeley Data | Source for the potato classes used in the integrated dataset | [Mendeley Data, DOI 10.17632/ptz377bwb8.1](https://data.mendeley.com/datasets/ptz377bwb8/1) |
| Disease Dataset of Wheat: Original, Augmented, and Balanced for Deep Learning | Mendeley Data | The 1,603-image original field dataset supplied the five wheat classes | [Mendeley Data, DOI 10.17632/5gc7hwydwg.1](https://data.mendeley.com/datasets/5gc7hwydwg/1) |

The exact Rice Leaf Diseases Kaggle archive downloaded for this work is named `Rice Leaf Diseases Dataset.zip` and contains eight Bangladeshi rice-leaf categories organized into training, validation, and testing directories. The linked Kaggle page matches the archive title and class structure. If the Kaggle owner changes or removes that version, use the archive title and class structure to locate the corresponding release.

## Final integrated classes

Only the following 15 categories were retained for the thesis experiments:

| Crop | Final repository classes |
|---|---|
| Potato | `potato_bacteria`, `potato_early_blight`, `potato_fungi`, `potato_healthy`, `potato_late_blight` |
| Rice | `rice_blast`, `rice_brown_spot`, `rice_healthy`, `rice_sheath_blight`, `rice_tungro_virus` |
| Wheat | `wheat_black_point`, `wheat_blast`, `wheat_fusarium_foot_rot`, `wheat_healthy`, `wheat_leaf_blight` |

The final integrated collection contains 6,551 images:

| Split | Images |
|---|---:|
| Training | 4,581 |
| Validation | 978 |
| Testing | 992 |
| **Total** | **6,551** |

## Integration process

1. Images were selected from the original crop-specific sources.
2. Disease names and folder structures were standardized into the 15 final class names.
3. Classes outside the thesis scope were excluded.
4. The combined image collection was divided into training, validation, and testing directories.
5. Images were resized to `224 x 224` at load time; the stored source images retain their original resolutions.
6. Data augmentation was applied during training only, not permanently written into `MasterDataset/`.

## Important evaluation note

The original split was performed at image level. A later audit found visually similar or duplicate captures across partitions, which can make the reported test accuracy optimistic as an estimate of performance on completely new field specimens. See [`AUDIT.md`](AUDIT.md) and [`audit_leakage_manifest.csv`](audit_leakage_manifest.csv) for the detailed analysis.

For future research, de-duplicate the original collection and create a seeded specimen- or capture-session-level split before retraining.

## Suggested acknowledgement

When building on this repository, cite the original source datasets directly and describe this repository's data as an integrated or derived multi-crop collection. A suitable sentence is:

> The experiments use an integrated multi-crop dataset assembled from publicly available rice, potato, and wheat disease image sources on Kaggle and Mendeley Data; the source links and transformation details are documented in `DATASETS.md`.

