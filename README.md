# Heart Reader — Multi-Label 12-Lead ECG Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/Edge-TFLite%20float16-00C853)
![Macro AUC](https://img.shields.io/badge/Best%20Macro%20AUC-0.9210-blueviolet)
![Macro F1](https://img.shields.io/badge/Best%20Macro%20F1-0.7242-blue)
![Models](https://img.shields.io/badge/Architectures-3-orange)
![License](https://img.shields.io/badge/License-MIT-green)


Automated multi-label diagnostic classification of 12-lead ECG signals into five cardiac superclasses using deep learning. Three architectures — ResNet1D, InceptionTime, and a patch-based Transformer — are trained, evaluated, and converted to TensorFlow Lite (float16) for edge deployment.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Preprocessing](#5-preprocessing)
6. [Model Architectures](#6-model-architectures)
7. [Training Details](#7-training-details)
8. [Evaluation & Results](#8-evaluation--results)
9. [Edge Optimization — TFLite](#9-edge-optimization--tflite)
10. [How to Reproduce](#10-how-to-reproduce)
11. [References](#11-references)

---

## 1. Problem Statement

The objective is to design a deep learning system that performs **multi-label classification** on 12-lead ECG recordings. Each recording may correspond to one or more of the following five diagnostic superclasses:

| Superclass | Description            |
| ---------- | ---------------------- |
| **NORM**   | Normal ECG             |
| **MI**     | Myocardial Infarction  |
| **STTC**   | ST/T Change            |
| **CD**     | Conduction Disturbance |
| **HYP**    | Hypertrophy            |

The primary metric is **Macro AUC** (or Macro F1-Score) on the held-out test set (Fold 10).

---

## 2. Dataset

### 2.1 Primary: PTB-XL v1.0.3

- **Source:** [PhysioNet — PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- **Records:** 21,799 twelve-lead ECG recordings from 18,869 patients
- **Sampling rate:** 100 Hz (downsampled; 1000 samples = 10 seconds per record)
- **Leads:** I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- **Signal shape per record:** `(1000, 12)` — 1000 timesteps × 12 leads

### 2.2 Extension: PTB-XL+ (ECGDeli Features)

- **Source:** [PhysioNet — PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- **Used features:** ECGDeli algorithm-derived structured features (fiducial points, intervals, amplitudes)
- **Purpose:** Augmented secondary input to improve model interpretability and performance

### 2.3 Data Split (Mandatory)

The stratified fold column (`strat_fold`) from `ptbxl_database.csv` is used:

| Split          | Folds | Samples |
| -------------- | ----- | ------- |
| **Train**      | 1–8   | ~17,441 |
| **Validation** | 9     | ~2,203  |
| **Test**       | 10    | ~2,155  |

> Deviation from this split is not permitted.

---

## 3. Project Structure

```
ECG/
├── README.md
├── notebooks/
│   ├── preprocessing.ipynb         ← Data loading, normalization, feature extraction
│   ├── train_resnet1d.ipynb        ← ResNet1D training + evaluation + TFLite export
│   ├── train_inception.ipynb       ← InceptionTime training + evaluation + TFLite export
│   ├── train_transformer.ipynb     ← ECG-Transformer training + evaluation + TFLite export
│   └── compare.ipynb               ← Side-by-side TFLite model comparison
│
└── models/
    ├── comparison.png                      ← Aggregate comparison plots
    ├── best_confusion_matrices.png         ← Best model confusion matrices
    ├── resnet1D/
    │   ├── best_ecg_model.keras            ← Keras checkpoint
    │   ├── ecg_model_fp16.tflite           ← TFLite float16
    │   ├── ecg_model_dynamic.tflite        ← TFLite dynamic range
    │   ├── training_curves.png
    │   ├── roc_curves.png
    │   └── confusion_matrices.png
    ├── inception/
    │   ├── best_inception_model.keras
    │   ├── inception_fp16.tflite
    │   ├── training_curves (1).png
    │   ├── roc_curves (1).png
    │   └── confusion_matrices (1).png
    └── transformer/
        ├── best_transformer_model.keras
        ├── transformer_fp16.tflite
        ├── training_curves.png
        ├── roc_curves.png
        └── confusion_matrices.png
```

---

## 4. Pipeline Overview

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  PTB-XL raw  │────▶│  preprocessing   │────▶│  Train notebook  │────▶│   TFLite     │
│  signals     │     │  .ipynb          │     │  (per arch)      │     │   .tflite    │
│  + PTB-XL+   │     │                  │     │                  │     │              │
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
                       │                        │                        │
                       ▼                        ▼                        ▼
                     • Load & parse labels    • Load preprocessed      • Float16 quant
                     • wfdb signal loading      arrays                 • Size ~50% of
                     • Per-lead z-norm        • FFT bandpass             Keras model
                     • PTB-XL+ ECGDeli        • Augmentation layer     • Validated on
                       feature extraction     • Weighted BCE loss        test set
                     • Save .npy arrays       • Train + evaluate
```

---

## 5. Preprocessing

**Notebook:** `notebooks/preprocessing.ipynb`

### 5.1 Label Extraction

- Parse `scp_codes` column from `ptbxl_database.csv` using the SCP statement reference table
- Map each diagnostic code to its superclass (NORM, MI, STTC, CD, HYP)
- Records can have **multiple labels** (multi-label)

### 5.2 Signal Loading

- Load all 21,799 ECG signals at 100 Hz using the `wfdb` library
- Each record: `(1000, 12)` float32 array

### 5.3 Normalization

- **Per-lead z-score standardization** using training set statistics:
  ```
  X_normalized = (X - mean_train) / std_train
  ```
- Statistics computed across all training samples and timesteps per lead
- Applied to train, validation, and test splits

### 5.4 PTB-XL+ Feature Integration

- Load ECGDeli features from PTB-XL+ extension
- Fill missing values with 0
- Normalize features with training-set mean/std
- Output: `F_train.npy`, `F_val.npy`, `F_test.npy`

### 5.5 Saved Outputs

Preprocessed data arrays produced by the preprocessing notebook:

| File                   | Shape                | Description                          |
| ---------------------- | -------------------- | ------------------------------------ |
| `X_train.npy`          | `(~17441, 1000, 12)` | Normalized ECG signals (train)       |
| `X_val.npy`            | `(~2203, 1000, 12)`  | Normalized ECG signals (val)         |
| `X_test.npy`           | `(~2155, 1000, 12)`  | Normalized ECG signals (test)        |
| `Y_train/val/test.npy` | `(N, 5)`             | Multi-label binary targets           |
| `F_train/val/test.npy` | `(N, F)`             | Normalized ECGDeli features          |
| `metadata.json`        | —                    | Superclass names, shapes, lead names |

---

## 6. Model Architectures

All three architectures share a **dual-input design**:

```
Signal Input (1000, 12)  ──▶  [Architecture-specific encoder]  ──▶  Signal embedding
                                                                          │
Feature Input (F,)  ──▶  [MLP: 128→64]  ──▶  Feature embedding           │
                                                    │                     │
                                              ┌─────┴─────────────────────┘
                                              ▼
                                        Concatenate
                                              │
                                        Dense(256) → BN → Dropout(0.4)
                                        Dense(128) → Dropout(0.3)
                                        Dense(5, sigmoid)
```

### 6.1 ResNet1D (`notebooks/train_resnet1d.ipynb`)

A 1D residual convolutional network adapted for ECG time series.

| Component          | Details                                                                            |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Stem**           | Conv1D(64, k=15) → BN → ReLU → MaxPool                                             |
| **Blocks**         | 7 residual blocks — filters: 64→64→128→128→256→256→512, kernels: 15→15→11→11→7→7→5 |
| **Residual block** | Conv1D → BN → ReLU → Dropout(0.1) → Conv1D → BN → Skip add → ReLU                  |
| **Pooling**        | MaxPool(2) after every 2 blocks, GlobalAveragePooling1D at the end                 |
| **Output**         | 512-dim embedding                                                                  |

### 6.2 InceptionTime (`notebooks/train_inception.ipynb`)

Multi-scale 1D CNN inspired by InceptionTime (Fawaz et al., 2020).

| Component              | Details                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Inception module**   | Bottleneck Conv1D(1) → 3 parallel Conv1D branches (k=10, 20, 40) + MaxPool branch → Concatenate → BN → ReLU |
| **Residual blocks**    | 3 blocks of 2 Inception modules each, with shortcut connections                                             |
| **Filter progression** | 32 → 64 → 128 filters per branch (×4 branches = 128 → 256 → 512 channels)                                   |
| **Pooling**            | MaxPool(2) between blocks, GAP at the end                                                                   |
| **Output**             | 512-dim embedding                                                                                           |

### 6.3 ECG-Transformer (`notebooks/train_transformer.ipynb`)

Patch-based Transformer encoder, inspired by ViT adapted for 1D signals.

| Component               | Details                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------- |
| **Patch Embedding**     | Conv1D(128, k=25, stride=25) → LayerNorm — 1000 timesteps → 40 patches              |
| **CLS Token**           | Learnable prepended token for global representation                                 |
| **Positional Encoding** | Learnable embeddings (max_len=41)                                                   |
| **Encoder**             | 4 Transformer blocks — d_model=128, 8 heads, FFN dim=256, pre-norm, GELU activation |
| **Output**              | CLS token → 128-dim embedding                                                       |
| **Optimizer**           | AdamW (weight_decay=1e-4) with cosine decay + 5-epoch linear warmup (peak LR=5e-4)  |

---

## 7. Training Details

### 7.1 Shared Training Enhancements

All three architectures use the following techniques:

#### Weighted Binary Cross-Entropy

Class imbalance is addressed by computing per-class pos-weights:

$$w_j = \frac{N_{\text{neg},j}}{N_{\text{pos},j}}$$

The weighted BCE loss up-weights minority positive samples:

$$\mathcal{L} = -\frac{1}{C} \sum_{j=1}^{5} \left[ w_j \cdot y_j \log(\hat{y}_j) + (1-y_j) \log(1-\hat{y}_j) \right]$$

#### FFT Bandpass Artifact Suppression

Applied once before training (deterministic preprocessing):

- **Passband:** 0.5–40 Hz (suppresses baseline wander < 0.5 Hz and high-frequency noise > 40 Hz)
- **Transition:** Tukey-style smooth rolloff over 0.5 Hz transition width to avoid spectral ringing
- **Implementation:** FFT → multiply frequency-domain mask → IFFT, applied per-lead

#### ECGAugmentation Layer (on-the-fly, training only)

A custom Keras layer applied inside the model graph, active only during `training=True`:

| Augmentation       | Parameters          | Purpose                                                                |
| ------------------ | ------------------- | ---------------------------------------------------------------------- |
| **Time masking**   | ratio=0.2, prob=0.8 | Zero out a contiguous 20% time segment (80% chance per sample)         |
| **Lead masking**   | prob=0.3            | Zero out random leads to simulate electrode dropout (max 50% of leads) |
| **Gaussian noise** | std=0.05            | Additive noise for regularization                                      |

### 7.2 Hyperparameters

| Parameter          | ResNet1D              | InceptionTime         | Transformer           |
| ------------------ | --------------------- | --------------------- | --------------------- |
| **Optimizer**      | Adam                  | Adam                  | AdamW                 |
| **Initial LR**     | 1e-3                  | 1e-3                  | 5e-4 (peak)           |
| **LR Schedule**    | ReduceLROnPlateau     | ReduceLROnPlateau     | Warmup + Cosine Decay |
| **Batch size**     | 64                    | 32\*                  | 64                    |
| **Max epochs**     | 50                    | 50                    | 60                    |
| **Early stopping** | patience=10 (val AUC) | patience=12 (val AUC) | patience=15 (val AUC) |
| **Checkpoint**     | Best val AUC          | Best val AUC          | Best val AUC          |

> \*InceptionTime uses batch size 32 to avoid GPU OOM on Colab.

### 7.3 Runtime

All models were trained on **Google Colab** with a single GPU (T4 / V100 / A100, depending on availability). Preprocessing is executed once; training notebooks load the resulting `.npy` arrays directly.

---

## 8. Evaluation & Results

### 8.1 Protocol

- Evaluation on **Fold 10** (test set, ~2155 samples)
- Metrics: **per-class AUC**, **per-class F1** (threshold=0.5), **Macro AUC**, **Macro F1**
- All models evaluated both as Keras checkpoints and after TFLite conversion

### 8.2 Performance Report

All TFLite (float16) models evaluated on the full test set (Fold 10). Ranked by Macro AUC:

| Model                | Macro AUC  | Macro F1   | Size (MB) | Latency (ms/sample) |
| -------------------- | ---------- | ---------- | --------- | ------------------- |
| **InceptionTime** ⭐ | **0.9210** | **0.7242** | 7.43      | 33.91               |
| **ResNet1D**         | 0.9140     | 0.7145     | 9.83      | 19.47               |
| **Transformer**      | 0.9018     | 0.6858     | 1.54      | 0.89                |

Per-class AUC breakdown:

| Model             | NORM   | MI     | STTC   | CD     | HYP    |
| ----------------- | ------ | ------ | ------ | ------ | ------ |
| **InceptionTime** | 0.9454 | 0.9096 | 0.9299 | 0.9181 | 0.9018 |
| **ResNet1D**      | 0.9432 | 0.9076 | 0.9276 | 0.8987 | 0.8929 |
| **Transformer**   | 0.9344 | 0.8743 | 0.9265 | 0.8914 | 0.8823 |

Per-class F1 breakdown (threshold = 0.5):

| Model             | NORM   | MI     | STTC   | CD     | HYP    |
| ----------------- | ------ | ------ | ------ | ------ | ------ |
| **InceptionTime** | 0.8443 | 0.7112 | 0.7526 | 0.7169 | 0.5958 |
| **ResNet1D**      | 0.8532 | 0.7127 | 0.7353 | 0.7007 | 0.5708 |
| **Transformer**   | 0.8309 | 0.6699 | 0.7329 | 0.6900 | 0.5051 |

### 8.3 Best Model — InceptionTime (Classification Report)

```
              precision    recall  f1-score   support

        NORM       0.76      0.95      0.84       963
          MI       0.62      0.84      0.71       550
        STTC       0.71      0.80      0.75       506
          CD       0.64      0.81      0.72       496
         HYP       0.52      0.71      0.60       262

   macro avg       0.65      0.82      0.72      2777
weighted avg       0.68      0.85      0.76      2777
```

### 8.4 Recommendation

| Criterion                        | Winner        | Details                                      |
| -------------------------------- | ------------- | -------------------------------------------- |
| **Best accuracy**                | InceptionTime | Macro AUC = 0.9210, Macro F1 = 0.7242        |
| **Best for edge**                | Transformer   | 1.54 MB, 0.89 ms/sample (Macro AUC = 0.9018) |
| **Best accuracy/size trade-off** | Transformer   | AUC/MB = 0.586                               |

### 8.5 Comparison Notebook

`notebooks/compare.ipynb` provides the full comparison pipeline. Visualizations generated include:

- Macro AUC & F1 grouped bar chart
- Per-class AUC grouped bars across all 3 models
- Mean ROC curves (averaged across 5 classes)
- Size vs. latency scatter plot (lower-left = better for edge)
- Confusion matrices for the best model (InceptionTime)

All plots are saved to `models/comparison.png` and `models/best_confusion_matrices.png`.

---

## 9. Edge Optimization — TFLite

### 9.1 Conversion Method

All models are converted to TensorFlow Lite using **float16 post-training quantization**:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

The ResNet1D model additionally has a **dynamic range quantization** variant (int8 weights, float activations).

### 9.2 Quantization Summary

| Technique         | Description                                         | Expected Compression   |
| ----------------- | --------------------------------------------------- | ---------------------- |
| **Float16**       | Weights stored in 16-bit float                      | ~50% size reduction    |
| **Dynamic range** | Weights quantized to int8, activations remain float | ~60–75% size reduction |

### 9.3 Model Artifacts

| Model         | Keras Checkpoint                                  | TFLite (float16)                             | TFLite (dynamic)                           |
| ------------- | ------------------------------------------------- | -------------------------------------------- | ------------------------------------------ |
| ResNet1D      | `models/resnet1D/best_ecg_model.keras`            | `models/resnet1D/ecg_model_fp16.tflite`      | `models/resnet1D/ecg_model_dynamic.tflite` |
| InceptionTime | `models/inception/best_inception_model.keras`     | `models/inception/inception_fp16.tflite`     | —                                          |
| Transformer   | `models/transformer/best_transformer_model.keras` | `models/transformer/transformer_fp16.tflite` | —                                          |

---

## 10. How to Reproduce

### 10.1 Requirements

- **Runtime:** Google Colab (GPU recommended) or any Python environment with GPU support
- **Python:** 3.10+
- **Key libraries:** TensorFlow ≥ 2.15, NumPy, pandas, matplotlib, seaborn, scikit-learn, wfdb, kagglehub

### 10.2 Step-by-Step

1. **Download the datasets:**
   - The notebooks use `kagglehub` to download PTB-XL and PTB-XL+ automatically.
   - Alternatively, download manually from [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/).

2. **Run preprocessing:**

   ```
   Open notebooks/preprocessing.ipynb → Run All
   ```

   Loads raw ECG data, extracts diagnostic labels, normalizes signals, integrates ECGDeli features, and saves preprocessed `.npy` arrays.

3. **Train models (run any or all):**

   ```
   Open notebooks/train_resnet1d.ipynb    → Run All  (ResNet1D)
   Open notebooks/train_inception.ipynb   → Run All  (InceptionTime)
   Open notebooks/train_transformer.ipynb → Run All  (ECG-Transformer)
   ```

   Each notebook:
   - Loads the preprocessed arrays
   - Applies FFT bandpass filtering
   - Defines the architecture with on-the-fly augmentation
   - Trains with weighted BCE and early stopping
   - Evaluates on the test set (Fold 10)
   - Converts the best checkpoint to TFLite (float16)
   - Saves all artifacts to `models/<architecture>/`

4. **Compare models:**
   ```
   Open notebooks/compare.ipynb → Run All
   ```
   Loads all three TFLite models, runs inference on the test set, and produces comparative metrics and visualizations.

---

## 11. References

1. Wagner, P., et al. (2020). _PTB-XL, a large publicly available electrocardiography dataset._ Scientific Data, 7(1), 154. [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Strodthoff, N., et al. (2023). _PTB-XL+, a comprehensive electrocardiographic feature dataset._ Scientific Data. [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/)
3. Fawaz, H. I., et al. (2020). _InceptionTime: Finding AlexNet for Time Series Classification._ Data Mining and Knowledge Discovery, 34, 1936–1962.
4. Dosovitskiy, A., et al. (2021). _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ ICLR 2021.
5. Natarajan, A., et al. (2020). _A Wide and Deep Transformer Neural Network for 12-Lead ECG Classification._ Computing in Cardiology.
6. He, K., et al. (2016). _Deep Residual Learning for Image Recognition._ CVPR 2016.

---

