# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

KAGGLE_DATA_ROOT = Path(
    "/kaggle/input/datasets/sakshamshukla191/sen2gf3floods/Sen2GF3Floods"
)

if not KAGGLE_DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Sen2GF3Floods directory not found: {KAGGLE_DATA_ROOT}"
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 100

# Per-GPU batch size (4 samples/GPU x 2 GPUs = effective batch 8)
BATCH_SIZE = 4

# Workers PER DDP PROCESS (2 GPUs x 2 workers = 4 workers total)
NUM_WORKERS = 2


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_ID = "nvidia/mit-b2"
NUM_CLASSES = 2
DECODER_DIM = 256


# ------------------------------------------------------------
# Loss (Tversky region term)
# ------------------------------------------------------------
# alpha weights false positives, beta weights false negatives.
# alpha > beta favors precision; alpha < beta favors recall.
# Test-set validation showed flood precision (0.883) lagging
# recall (0.918) and lagging the paper's U-Net++ baseline
# precision (0.959), so alpha > beta is used to correct for it.

REGION_LOSS_MODE = "tversky"
REGION_LOSS_WEIGHT = 0.5
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3

# ------------------------------------------------------------
# Loss (weighted cross-entropy flood weight cap)
# ------------------------------------------------------------
# Reverted to the confirmed Tversky baseline (4.0, non-binding
# since the natural value is ~3.722) for this experiment. This
# run's only new variable is the architecture change below --
# the flood-weight-cap experiment is a separate, independent test
# and is deliberately NOT stacked on top of this one, so any
# result difference can be attributed to the architecture alone.

FLOOD_WEIGHT_CAP = 4.0

# ------------------------------------------------------------
# High-resolution decoder branch
# ------------------------------------------------------------
# Error decomposition on the Tversky model showed ~90%+ of FN
# pixels within 10px of the ground-truth flood boundary, and
# several validation images with 60-75% of a genuine flood region
# missed -- both consistent with the decoder losing fine spatial
# detail once features are compressed to the 64x64 stage-0
# resolution. This branch is a small, shallow conv stack that
# reads the raw 6-channel input at full 256x256 resolution and
# feeds learned fine-detail features into the decoder alongside
# the four (bilinearly upsampled) encoder scales, so the final
# per-pixel decision is made with genuine full-resolution
# information available, not just decided at 64x64 and upsampled.

HIGH_RES_BRANCH_HIDDEN = 32
HIGH_RES_BRANCH_OUT = 64


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

CHECKPOINT_DIR = Path(
    "/kaggle/working/flood_model_checkpoints"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
