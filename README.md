# Multimodal Flood Segmentation Using Dual-Stream SegFormer

A deep learning project for pixel-level flood segmentation using Sentinel-2 optical imagery and Gaofen-3 (GF-3) Synthetic Aperture Radar (SAR) data.

This project implements a **dual-stream, cross-modal SegFormer architecture** with modality-specific feature extraction, cross-modal fusion, and a high-resolution feature branch. The model is trained and evaluated on the Sen2GF3Floods dataset.

The objective is to combine complementary optical and SAR information to identify flooded regions in satellite imagery.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Input Data](#input-data)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Checkpointing and Resuming](#checkpointing-and-resuming)
- [Limitations](#limitations)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

Flood mapping from satellite imagery is an important remote-sensing task. Accurate segmentation requires identifying flooded pixels while distinguishing them from non-flooded regions, including existing water bodies and other visually similar surfaces.

Optical and SAR imagery provide different types of information:

- **Optical imagery** captures reflected sunlight across spectral bands and provides information about surface characteristics.
- **SAR imagery** measures microwave backscatter and can provide information under cloudy conditions and at night.

The Sen2GF3Floods dataset combines pre-disaster Sentinel-2 optical imagery with post-disaster GF-3 SAR imagery. It contains 21,483 samples covering nine major flood events. [1]

This project explores multimodal learning by processing the optical and SAR inputs through separate feature-extraction streams and combining their learned representations for flood segmentation.

The segmentation task is formulated as a binary pixel-classification problem:

| Class | Meaning |
|---|---|
| 0 | Background / non-flood |
| 1 | Flood |

The model predicts a class for every pixel in the input image.

---

## Objectives

The main objectives of this project are:

1. Develop a dual-stream architecture for separate optical and SAR feature extraction.
2. Adapt a pretrained SegFormer MiT-B2 encoder for the six-channel multimodal input.
3. Explore gated fusion and cross-attention for combining features from the two modalities.
4. Incorporate high-resolution image features to preserve spatial details relevant to flood boundaries.
5. Address class imbalance using weighted cross-entropy and Tversky loss.
6. Train and evaluate the model using a separate training, validation, and test split.
7. Support mixed-precision training, multi-GPU Distributed Data Parallel (DDP), and resumable checkpoints.

---

## Dataset

### Sen2GF3Floods

This project uses the **Sen2GF3Floods** multimodal remote-sensing dataset.

- Dataset: [Sen2GF3Floods — Science Data Bank](https://doi.org/10.57760/sciencedb.25341)
- Dataset paper: [Sen2GF3Floods: A Benchmark Multi-Source Flood Dataset with Dual-Temporal and Active Learning Annotation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13062051/)

The dataset contains 21,483 standardized samples from nine major flood events. Each sample combines four Sentinel-2 optical bands and two GF-3 SAR polarization channels. The dataset documentation reports a common spatial resolution of 10 metres. [1]

The dataset is organized into three main directories:

```text
Sen2GF3Floods/
├── sentinel2/
├── gaofen3/
└── label/
```

- `sentinel2/`: Sentinel-2 optical imagery
- `gaofen3/`: GF-3 SAR imagery
- `label/`: Flood segmentation masks

The dataset is **not included in this repository**. Users must obtain it separately and configure the dataset path.

---

## Input Data

Each input sample contains six channels with a spatial size of 256 × 256 pixels.

| Channel | Source | Description |
|---|---|---|
| B4 | Sentinel-2 | Red |
| B3 | Sentinel-2 | Green |
| B2 | Sentinel-2 | Blue |
| B8 | Sentinel-2 | Near-infrared (NIR) |
| HH | GF-3 | Horizontal transmit, horizontal receive polarization |
| HV | GF-3 | Horizontal transmit, vertical receive polarization |

Sentinel-2 provides multispectral optical observations, while GF-3 supplies SAR measurements. [1, 2]

The input tensor has the shape:

```text
[6, 256, 256]
```

The corresponding segmentation mask has the shape:

```text
[256, 256]
```

The model produces two class logits for each pixel:

```text
[2, 256, 256]
```

The optical and SAR channels are handled by separate branches in the model rather than being processed by a single shared input projection.

---

## Model Architecture

### Dual-Stream Cross-Modal SegFormer

The model is based on SegFormer, a semantic-segmentation architecture that uses a hierarchical Transformer encoder and a lightweight decoder to aggregate multiscale features. [3]

The implementation uses the pretrained **MiT-B2** backbone from the Hugging Face Transformers library.

### 1. Optical stream

The optical stream processes the four Sentinel-2 channels:

```text
B4, B3, B2, B8
```

Its purpose is to learn features from optical and spectral information.

### 2. SAR stream

The SAR stream processes the two GF-3 channels:

```text
HH, HV
```

It learns features from SAR backscatter and polarization information.

### 3. Pretrained MiT-B2 encoders

The model loads a pretrained SegFormer MiT-B2 model and creates separate optical and SAR encoders.

The initial patch-embedding projections are adapted to accept four optical channels and two SAR channels, respectively.

This allows the two streams to learn modality-specific representations while benefiting from pretrained encoder weights.

### 4. Cross-modal feature fusion

The two streams are combined at multiple feature stages.

The implementation uses:

- **Gated fusion** in the earlier stages.
- **Bidirectional cross-attention** in the later stages.
- Feature-merging convolutional layers after fusion.

Gated fusion learns how much information to use from each modality. Cross-attention allows features from one modality to interact with features from the other modality.

This is intended to let the network use complementary optical and SAR information rather than treating all six channels as interchangeable inputs.

### 5. High-resolution branch

The model includes a separate convolutional branch that processes the original six-channel input at full spatial resolution.

Its purpose is to preserve fine spatial details that may be lost when the encoder progressively reduces the spatial resolution of its feature maps.

The high-resolution features are combined with the multiscale encoder features in the decoder.

### 6. Segmentation decoder

The decoder combines the fused multiscale features and high-resolution features to produce a two-class segmentation map.

The output is a per-pixel prediction of:

- Background
- Flood

### Architecture summary

```text
                 Six-channel input
                 256 × 256 × 6
                        |
          +-------------+-------------+
          |                           |
    Optical input                 SAR input
    4 channels                   2 channels
          |                           |
    MiT-B2 encoder               MiT-B2 encoder
          |                           |
          +------ Multimodal fusion --+
                 Gated fusion
                 Cross-attention
                        |
              Multiscale features
                        |
Six-channel input --> High-resolution branch
                        |
                 Feature decoder
                        |
                Segmentation head
                        |
             Background / Flood map
```

The diagram is a conceptual summary of the implementation. See `model8.py` for the exact module definitions and tensor operations.

---

## Loss Function

Flood segmentation is affected by class imbalance because flood pixels may represent a relatively small proportion of an image.

This implementation combines **weighted cross-entropy loss** and **Tversky loss**.

The total loss is:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{WCE}}
+
\lambda_{\mathrm{region}}
\mathcal{L}_{\mathrm{Tversky}}
\]

where:

- \(\mathcal{L}_{\mathrm{WCE}}\) is weighted cross-entropy loss.
- \(\mathcal{L}_{\mathrm{Tversky}}\) is the region-based Tversky loss.
- \(\lambda_{\mathrm{region}}\) controls the contribution of the region loss.

### Weighted cross-entropy

Weighted cross-entropy assigns different weights to the background and flood classes. The class weights are calculated from the training-set class distribution, with a configurable cap on the flood-class weight.

This helps prevent the majority background class from dominating the loss.

### Tversky loss

The Tversky index is:

\[
TI =
\frac{TP}
{TP+\alpha FP+\beta FN}
\]

The corresponding loss is:

\[
\mathcal{L}_{\mathrm{Tversky}}=1-TI
\]

where:

- \(TP\): true-positive flood pixels
- \(FP\): false-positive flood pixels
- \(FN\): false-negative flood pixels
- \(\alpha\): weight assigned to false positives
- \(\beta\): weight assigned to false negatives

The configuration uses:

```python
REGION_LOSS_MODE = "tversky"
REGION_LOSS_WEIGHT = 0.5

TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
```

Since \(\alpha > \beta\), the Tversky term assigns a greater penalty to false positives than to false negatives, all else being equal.

The loss implementation is in `losses8.py`, while its configuration is in `config8.py`.

---

## Training Strategy

The training pipeline includes the following components:

### Data preparation

- Load optical imagery, SAR imagery, and segmentation labels.
- Create training, validation, and test subsets.
- Compute channel statistics using the training data.
- Normalize the input channels.
- Apply training-time spatial augmentations.
- Calculate class weights from training-set class counts.

The configured split ratios are:

| Split | Proportion |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

A fixed random seed is used for reproducibility of the split and relevant random operations.

### Optimizer

The implementation uses AdamW with discriminative learning rates:

| Parameter group | Learning rate |
|---|---:|
| Optical and SAR encoders | \(1 \times 10^{-5}\) |
| Fusion and decoder layers | \(1 \times 10^{-4}\) |

The optimizer uses a weight decay of \(10^{-2}\).

### Learning-rate scheduler

The learning-rate schedule consists of:

1. Linear warm-up for the first five epochs.
2. Cosine annealing for the remaining epochs.

The minimum learning rate is configured as \(10^{-6}\).

### Mixed precision

Automatic mixed precision (AMP) is enabled during training and evaluation to reduce memory usage and improve computational efficiency on supported NVIDIA GPUs.

Gradient clipping is also used, with a maximum gradient norm of 1.0.

### Distributed training

The training utilities support PyTorch Distributed Data Parallel (DDP).

When launched with `torchrun`, each process is assigned a GPU and participates in distributed training. The effective batch size is:

\[
B_{\mathrm{effective}}
=
B_{\mathrm{GPU}}\times N_{\mathrm{GPUs}}
\]

With the default configuration of four samples per GPU and two GPUs, the effective batch size is eight.

If the script is run without `torchrun`, it falls back to a single-process run on GPU 0.

---

## Evaluation Metrics

The model is evaluated using the following metrics.

### Intersection over Union (IoU)

For a class \(c\):

\[
IoU_c =
\frac{TP_c}
{TP_c+FP_c+FN_c}
\]

### Mean Intersection over Union (mIoU)

For the two classes:

\[
mIoU =
\frac{IoU_{\mathrm{background}}
+IoU_{\mathrm{flood}}}{2}
\]

### Precision

\[
Precision =
\frac{TP}{TP+FP}
\]

### Recall

\[
Recall =
\frac{TP}{TP+FN}
\]

### F1-score

\[
F1 =
2\frac{Precision \times Recall}
{Precision+Recall}
\]

The training pipeline tracks validation loss, accuracy, precision, recall, F1-score, and mIoU. The best checkpoint is selected using validation mIoU, and the test set is evaluated using the selected checkpoint.

---

## Experimental Results

The following results were obtained in the project’s high-resolution model experiment. They are included as the reported results of that run, rather than as a guarantee that another environment or training run will reproduce them.

### Test-set results

| Metric | Result |
|---|---:|
| Mean IoU (mIoU) | 91.4616% |
| Flood IoU | 84.1417% |
| Background IoU | 98.7816% |
| Precision | 90.9057% |
| Recall | 91.8754% |
| F1-score | 91.3880% |

These results use the model's standard `argmax` predictions.

### Threshold-based evaluation

A separate validation-based threshold experiment selected a flood-probability threshold of 0.64. Applying that threshold to the test set produced:

| Metric | Result |
|---|---:|
| Mean IoU (mIoU) | 91.5242% |
| Flood IoU | 84.2379% |
| Precision | 92.4651% |
| Recall | 90.4466% |
| F1-score | 91.4447% |

The threshold was selected using validation data, not by optimizing on the test set. The standard training script evaluates using `argmax`; reproducing the threshold-based experiment requires a separate probability-threshold evaluation.

---

## Project Structure

The repository contains six Python modules.

```text
.
├── config8.py
├── dataset8.py
├── losses8.py
├── model8.py
├── training8.py
├── model_train_ddp8.py
└── README.md
```

### `config8.py`

Contains the configuration for:

- Dataset location
- Number of epochs
- Batch size and data-loader workers
- Model identifier and decoder dimensions
- Loss-function parameters
- High-resolution branch dimensions
- Checkpoint directory

### `dataset8.py`

Implements:

- Sen2GF3Floods loading
- Optical and SAR data handling
- Segmentation-mask loading
- Dataset splitting
- Training-set channel statistics
- Class-weight calculation
- Normalization and augmentation
- Data loaders and distributed sampling

### `losses8.py`

Implements:

- Tversky loss
- Composite weighted cross-entropy and region loss

### `model8.py`

Implements:

- Pretrained MiT-B2 encoder loading
- Optical and SAR encoder streams
- Adapted input projections
- Gated feature fusion
- Cross-attention fusion
- High-resolution feature branch
- Segmentation decoder

### `training8.py`

Implements:

- Distributed training utilities
- Mixed-precision training
- Validation and test evaluation
- Metric calculation
- Checkpoint saving and loading
- Training resumption

### `model_train_ddp8.py`

The main training script. It:

1. Initializes the training environment.
2. Prepares the data loaders.
3. Creates the model and loss function.
4. Builds the optimizer and learning-rate scheduler.
5. Runs training and validation.
6. Loads the best validation checkpoint.
7. Evaluates the model on the test set.

---

## Requirements

The project uses Python and the following main libraries:

- PyTorch
- NumPy
- GDAL (`osgeo.gdal`)
- Hugging Face Transformers
- tqdm

A CUDA-enabled PyTorch installation and compatible NVIDIA GPU are required by the current training script.

Install the Python dependencies in an environment compatible with your CUDA and GDAL installations. For example:

```bash
pip install numpy tqdm transformers
```

Install the appropriate PyTorch build from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

GDAL must also be installed with Python bindings available as:

```python
from osgeo import gdal
```

The exact PyTorch, CUDA, Transformers, and GDAL versions should be recorded for a reproducible experiment. The code does not currently provide a pinned `requirements.txt`.

---

## Dataset Setup

The configuration currently uses a Kaggle-specific dataset path:

```python
KAGGLE_DATA_ROOT = Path(
    "/kaggle/input/datasets/sakshamshukla191/sen2gf3floods/Sen2GF3Floods"
)
```

Before running the project:

1. Obtain the Sen2GF3Floods dataset from its official source.
2. Make sure the dataset contains the expected `sentinel2`, `gaofen3`, and `label` directories.
3. Update `KAGGLE_DATA_ROOT` in `config8.py` to the location of the dataset in your environment.
4. Ensure the configured directory is accessible.

The current configuration checks whether the dataset path exists when `config8.py` is imported.

---

## Running the Project

### 1. Place the modules together

Keep the six Python files in the same directory.

The main script imports the modules using the unnumbered names:

```python
from config import ...
from dataset import ...
from model import ...
from losses import ...
from training import ...
```

Therefore, if the files in your repository are named `config8.py`, `dataset8.py`, and so on, rename them to the corresponding import names before running the script, or update the imports consistently.

For the commands below, the files are assumed to be named:

```text
config.py
dataset.py
losses.py
model.py
training.py
model_train_ddp.py
```

### 2. Configure the dataset path

Update `KAGGLE_DATA_ROOT` in `config.py` to match your dataset location.

### 3. Run on one GPU

From the project directory:

```bash
python model_train_ddp.py
```

The script requires CUDA. Running without `torchrun` uses one GPU, even if multiple GPUs are visible.

### 4. Run on two GPUs

To use two GPUs through DDP:

```bash
torchrun --standalone --nproc_per_node=2 model_train_ddp.py
```

Change `--nproc_per_node` to the number of GPUs you intend to use.

The configured batch size is per GPU, so the effective batch size changes with the number of processes.

### 5. Pretrained weights

The model uses the Hugging Face model identifier:

```text
nvidia/mit-b2
```

The pretrained model must be available through the Hugging Face cache or downloadable from the Hugging Face Hub. A first run may require network access to retrieve the weights.

---

## Checkpointing and Resuming

Checkpoints are saved under the configured directory:

```text
/kaggle/working/flood_model_checkpoints/
```

The training script creates a model-specific subdirectory containing files such as:

```text
checkpoint_best.pth
checkpoint_latest.pth
checkpoint_epoch_005.pth
checkpoint_epoch_010.pth
...
```

- `checkpoint_best.pth`: checkpoint with the best validation mIoU.
- `checkpoint_latest.pth`: most recent training checkpoint.
- `checkpoint_epoch_XXX.pth`: periodic checkpoint.

Training is configured with `resume_training=True`. If a compatible latest checkpoint is found, the script restores the model, optimizer, scheduler, and other saved training state and continues from the next epoch.

**Important:** If you want to start a completely new experiment, use a fresh checkpoint directory or remove the previous run's checkpoints. Otherwise, the script may resume an existing run.

---

## Limitations

- The model is trained on the Sen2GF3Floods dataset; performance on other geographic regions, flood events, sensors, or acquisition conditions requires separate evaluation.
- The model performs binary segmentation and does not directly estimate flood depth, water velocity, or the time of inundation.
- Optical and SAR observations contain different information and may be affected by acquisition timing, registration, and environmental conditions.
- A segmentation model can confuse floodwater with permanent water bodies or other surfaces with similar image characteristics.
- Reported metrics depend on the dataset split, preprocessing, checkpoint, and evaluation procedure.
- The threshold-based results are distinct from the standard `argmax` evaluation.
- Multi-GPU training requires a correctly configured CUDA environment and a `torchrun` launch.

---

## References

**[1] Sen2GF3Floods dataset**

Mo, G., Xing, Z. *Sen2GF3Floods: A Novel Multi-Source Remote Sensing Dataset for Flood Mapping.*

Science Data Bank.

https://doi.org/10.57760/sciencedb.25341

Dataset article:

https://pmc.ncbi.nlm.nih.gov/articles/PMC13062051/

**[2] Sentinel-2 mission**

European Space Agency (ESA). *Sentinel-2.*

https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2

**[3] SegFormer**

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).

*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.*

Advances in Neural Information Processing Systems (NeurIPS 2021).

https://arxiv.org/abs/2105.15203

**[4] Hugging Face SegFormer documentation**

Hugging Face Transformers. *SegFormer model documentation.*

https://huggingface.co/docs/transformers/model_doc/segformer

---

## Acknowledgements

This project uses the Sen2GF3Floods dataset and builds upon the SegFormer architecture.

The dataset creators and the authors of SegFormer are acknowledged for making their research and resources available.

This repository contains the implementation developed for this project. Please consult the original dataset and model references for their respective terms of use and licensing.

