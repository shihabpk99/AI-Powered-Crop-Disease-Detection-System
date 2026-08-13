# AI-Powered Crop Disease Detection for Sustainable Agriculture in Bangladesh

This repository contains the implementation and evaluation of my undergraduate thesis on automatic crop disease classification from leaf images. The project focuses on three economically important crops in Bangladesh - **rice, potato, and wheat** - and uses deep learning to identify **15 healthy and diseased crop classes**.

The main contribution is a **feature-level fusion model** that combines the internal feature representations learned by DenseNet121, MobileNetV2, and EfficientNetB0. On the thesis test set, this final model achieved **96.27% accuracy**, with **0.97 precision**, **0.96 recall**, and a **0.96 F1-score**.

![Model accuracy comparison](Result%20Images/accuracy_comparison_final.png)

## Project Overview

Crop diseases can significantly reduce agricultural production, while manual diagnosis is often slow and dependent on expert availability. This research investigates whether transfer learning and ensemble learning can provide an effective image-based disease identification system for crops relevant to Bangladesh.

Given a leaf image, the system:

1. Resizes the image to `224 x 224` pixels.
2. Extracts visual features using pretrained convolutional neural networks.
3. Classifies the image into one of 15 crop-condition classes.
4. Returns the predicted disease or healthy category.

## Supported Classes

| Crop | Classes |
|---|---|
| **Potato** | Bacteria, Early Blight, Fungi, Healthy, Late Blight |
| **Rice** | Blast, Brown Spot, Healthy, Sheath Blight, Tungro Virus |
| **Wheat** | Black Point, Blast, Fusarium Foot Rot, Healthy, Leaf Blight |

## Dataset

The integrated dataset contains **6,551 field and uncontrolled-environment leaf images** collected from publicly available Kaggle and Mendeley Data sources. It was divided using a stratified **70% training, 15% validation, and 15% testing** split.

| Split | Images |
|---|---:|
| Training | 4,581 |
| Validation | 978 |
| Testing | 992 |
| **Total** | **6,551** |

To improve generalization, training images were augmented using rotation, translation, zoom, shear, brightness variation, and horizontal flipping. Validation and test images were not augmented. Class weighting was used to reduce bias caused by unequal class sizes.

## Methodology

The study was completed in four stages:

### 1. Transfer Learning

Three ImageNet-pretrained CNN architectures were used as fixed feature extractors:

- DenseNet121
- MobileNetV2
- EfficientNetB0

Their original classification layers were replaced with a new 15-class Softmax classifier.

### 2. Fine-Tuning

The upper layers of each backbone were unfrozen and trained with a small learning rate, allowing the networks to adapt their high-level visual features to crop disease symptoms.

### 3. Decision-Level Fusion

The prediction vectors from the three fine-tuned models were combined and passed through a learned classifier. This tested whether combining final model decisions could outperform the individual networks.

### 4. Feature-Level Fusion - Final Model

The final approach extracts the internal deep features of all three fine-tuned networks:

- DenseNet121: 1,024-dimensional features
- MobileNetV2: 1,280-dimensional features
- EfficientNetB0: 1,280-dimensional features

Each feature stream is compressed to 256 dimensions, then the three streams are concatenated into a 768-dimensional representation. A dense classification head uses this combined representation to predict the final class.

This approach performed best because it combines complementary texture, shape, edge, and spatial information before the final classification decision.

## Evaluation Results

All models were evaluated on the same 992-image test set using accuracy, precision, recall, F1-score, and confusion matrices.

| Model | Stage | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| DenseNet121 | Feature extraction | 90.93% | 0.91 | 0.91 | 0.91 |
| MobileNetV2 | Feature extraction | 89.92% | 0.90 | 0.90 | 0.90 |
| EfficientNetB0 | Feature extraction | 92.44% | 0.93 | 0.92 | 0.92 |
| DenseNet121 | Fine-tuned | 92.54% | 0.93 | 0.93 | 0.93 |
| MobileNetV2 | Fine-tuned | 93.65% | 0.94 | 0.94 | 0.94 |
| EfficientNetB0 | Fine-tuned | 93.15% | 0.94 | 0.93 | 0.93 |
| Decision-Level Fusion | Ensemble | 91.23% | 0.92 | 0.91 | 0.91 |
| **Feature-Level Fusion** | **Final model** | **96.27%** | **0.97** | **0.96** | **0.96** |

The best individual model was fine-tuned MobileNetV2 at 93.65%. The final feature-level fusion model improved the test accuracy to 96.27%, making it the selected architecture for the proposed system.

![Feature-level fusion confusion matrix](Result%20Images/Final_Experiment/Feature_Level_Fusion_Confusion_Matrix.png)

## Repository Structure

```text
.
|-- Final_attempt.ipynb                 # Main training and fusion pipeline
|-- Final_Evaluation_on_test.ipynb      # Final test-set evaluation
|-- MasterDataset/                      # Train, validation, and test images
|-- FinalModels/                        # Exported trained models
|-- Result Images/                      # Performance charts and confusion matrices
|-- check_setup.py                      # Environment verification script
|-- AUDIT.md                            # Additional methodological review
`-- README.md
```

The earlier experiment notebooks are retained as research history, but `Final_attempt.ipynb` and `Final_Evaluation_on_test.ipynb` contain the final workflow and reported evaluation.

## Getting Started

### Requirements

- Python 3.9
- TensorFlow 2.x with Keras 2
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV

A compatible environment can be created with:

```bash
conda create -n crop-disease python=3.9
conda activate crop-disease
pip install "tensorflow<2.16" numpy pandas matplotlib seaborn scikit-learn opencv-python split-folders
```

Check the environment:

```bash
python check_setup.py
```

Then open and run:

1. `Final_attempt.ipynb` for training and fusion model creation.
2. `Final_Evaluation_on_test.ipynb` for final evaluation and confusion matrices.

> **Note:** The notebooks were developed with local Windows paths. Update the dataset and model-output paths to match your machine before running them. Training from scratch is computationally expensive, so a CUDA-capable GPU is recommended.

## Final Model Artifacts

The `FinalModels/` directory contains the trained base models, fine-tuned models, fusion models, and a TensorFlow Lite export. The final research model is the feature-level fusion architecture, represented by the exported fusion artifacts in this directory.

## Research Note

The table above reports the results presented in the final thesis evaluation. For transparency, [`AUDIT.md`](AUDIT.md) documents a later methodological review, including dataset near-duplicate findings and reproducibility limitations that should be considered when interpreting the reported test accuracy or extending this research.

## Future Work

- Rebuild the dataset using specimen-level deduplication and splitting.
- Validate the system with newly collected images from farms in Bangladesh.
- Add more crops and disease classes.
- Optimize the model for mobile and offline inference.
- Build a farmer-friendly application with disease information and treatment guidance.

## Thesis

**Title:** *AI-Powered Crop Disease Detection System for Sustainable Agriculture in Bangladesh*

This repository accompanies the thesis and demonstrates how transfer learning and feature-level deep model fusion can be applied to multi-crop disease classification under locally relevant agricultural conditions.

## License

No license has been added yet. Please contact the repository owner before reusing the dataset, models, or source code.
