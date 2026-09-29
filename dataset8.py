# ============================================================
# DATASET AND DATA LOADING
# Sen2GF3Floods
# ============================================================

import math
import os
import re

import numpy as np
import torch
import torch.distributed as dist

from osgeo import gdal
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    random_split,
)

from config import (
    KAGGLE_DATA_ROOT,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    FLOOD_WEIGHT_CAP,
)


# ============================================================
# DATASET PATHS
# ============================================================

DATA_ROOT = str(KAGGLE_DATA_ROOT)

S2_DIR = os.path.join(DATA_ROOT, "sentinel2")
GF3_DIR = os.path.join(DATA_ROOT, "gaofen3")
LABEL_DIR = os.path.join(DATA_ROOT, "label")


# ============================================================
# HELPER
# ============================================================

def get_sample_id(filename):
    match = re.search(
        r"_(\d+)\.tiff?$",
        filename,
        re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not determine sample ID from: {filename}"
        )

    return int(match.group(1))


# ============================================================
# SEN2GF3FLOODS DATASET
# ============================================================

class Sen2GF3FloodsDataset(Dataset):

    def __init__(
        self,
        s2_dir,
        gf3_dir,
        label_dir,
    ):
        self.s2_dir = s2_dir
        self.gf3_dir = gf3_dir
        self.label_dir = label_dir

        self.s2_files = {}
        self.hh_files = {}
        self.hv_files = {}
        self.label_files = {}

        # ----------------------------------------------------
        # Sentinel-2
        # ----------------------------------------------------

        for filename in os.listdir(s2_dir):

            if filename.lower().endswith((".tif", ".tiff")):

                sample_id = get_sample_id(filename)

                self.s2_files[sample_id] = os.path.join(
                    s2_dir,
                    filename,
                )

        # ----------------------------------------------------
        # GF-3 HH
        # ----------------------------------------------------

        for filename in os.listdir(gf3_dir):

            if (
                "hh" in filename.lower()
                and filename.lower().endswith(
                    (".tif", ".tiff")
                )
            ):

                sample_id = get_sample_id(filename)

                self.hh_files[sample_id] = os.path.join(
                    gf3_dir,
                    filename,
                )

        # ----------------------------------------------------
        # GF-3 HV
        # ----------------------------------------------------

        for filename in os.listdir(gf3_dir):

            if (
                "hv" in filename.lower()
                and filename.lower().endswith(
                    (".tif", ".tiff")
                )
            ):

                sample_id = get_sample_id(filename)

                self.hv_files[sample_id] = os.path.join(
                    gf3_dir,
                    filename,
                )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        for filename in os.listdir(label_dir):

            if filename.lower().endswith((".tif", ".tiff")):

                sample_id = get_sample_id(filename)

                self.label_files[sample_id] = os.path.join(
                    label_dir,
                    filename,
                )

        # ----------------------------------------------------
        # Complete samples only
        # ----------------------------------------------------

        all_ids = (
            set(self.s2_files.keys())
            & set(self.hh_files.keys())
            & set(self.hv_files.keys())
            & set(self.label_files.keys())
        )

        self.sample_ids = sorted(all_ids)

        # ----------------------------------------------------
        # Report incomplete samples
        # ----------------------------------------------------

        all_possible_ids = (
            set(self.s2_files.keys())
            | set(self.hh_files.keys())
            | set(self.hv_files.keys())
            | set(self.label_files.keys())
        )

        missing_ids = sorted(
            all_possible_ids - set(self.sample_ids)
        )

        if missing_ids:
            print(
                f"Warning: {len(missing_ids)} sample IDs "
                "are missing one or more files."
            )
            print(
                "First missing IDs:",
                missing_ids[:10],
            )

        print(f"Complete samples found: {len(self.sample_ids)}")

    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    def __len__(self):
        return len(self.sample_ids)

    # --------------------------------------------------------
    # Read one raster band
    # --------------------------------------------------------

    @staticmethod
    def read_single_band(filepath):

        src = gdal.Open(filepath)

        if src is None:
            raise FileNotFoundError(
                f"Could not open: {filepath}"
            )

        data = src.GetRasterBand(1).ReadAsArray()

        return data.astype(np.float32)

    # --------------------------------------------------------
    # Read Sentinel-2
    # --------------------------------------------------------

    @staticmethod
    def read_sentinel2(filepath):

        src = gdal.Open(filepath)

        if src is None:
            raise FileNotFoundError(
                f"Could not open: {filepath}"
            )

        if src.RasterCount != 4:
            raise ValueError(
                f"Expected 4 Sentinel-2 bands, "
                f"found {src.RasterCount} in {filepath}"
            )

        # TIFF order:
        # Band 1 -> B2
        # Band 2 -> B3
        # Band 3 -> B4
        # Band 4 -> B8
        #
        # Return paper order:
        # B4, B3, B2, B8

        b2 = src.GetRasterBand(1).ReadAsArray().astype(
            np.float32
        )

        b3 = src.GetRasterBand(2).ReadAsArray().astype(
            np.float32
        )

        b4 = src.GetRasterBand(3).ReadAsArray().astype(
            np.float32
        )

        b8 = src.GetRasterBand(4).ReadAsArray().astype(
            np.float32
        )

        return b4, b3, b2, b8

    # --------------------------------------------------------
    # Get one sample
    # --------------------------------------------------------

    def __getitem__(self, index):

        sample_id = self.sample_ids[index]

        s2_file = self.s2_files[sample_id]
        hh_file = self.hh_files[sample_id]
        hv_file = self.hv_files[sample_id]
        label_file = self.label_files[sample_id]

        # Sentinel-2
        b4, b3, b2, b8 = self.read_sentinel2(
            s2_file
        )

        # GF-3
        hh = self.read_single_band(hh_file)
        hv = self.read_single_band(hv_file)

        # Label
        mask = self.read_single_band(
            label_file
        ).astype(np.int64)

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        expected_shape = (256, 256)

        for name, array in [
            ("B4", b4),
            ("B3", b3),
            ("B2", b2),
            ("B8", b8),
            ("HH", hh),
            ("HV", hv),
        ]:

            if array.shape != expected_shape:
                raise ValueError(
                    f"Sample {sample_id}: "
                    f"{name} has shape {array.shape}, "
                    f"expected {expected_shape}"
                )

        if mask.shape != expected_shape:
            raise ValueError(
                f"Sample {sample_id}: "
                f"mask has shape {mask.shape}, "
                f"expected {expected_shape}"
            )

        # ----------------------------------------------------
        # Six-channel image
        #
        # B4, B3, B2, B8, HH, HV
        # ----------------------------------------------------

        image = np.stack(
            [
                b4,
                b3,
                b2,
                b8,
                hh,
                hv,
            ],
            axis=0,
        )

        image = torch.from_numpy(
            image.astype(np.float32)
        )

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


# ============================================================
# CREATE BASE DATASET
# ============================================================

def create_dataset():

    return Sen2GF3FloodsDataset(
        s2_dir=S2_DIR,
        gf3_dir=GF3_DIR,
        label_dir=LABEL_DIR,
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_dataset_splits(dataset, seed=SEED):

    generator = torch.Generator().manual_seed(seed)

    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = (
        total_size
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [
            train_size,
            val_size,
            test_size,
        ],
        generator=generator,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# TRAIN-ONLY CHANNEL STATISTICS
# ============================================================

@torch.no_grad()
def compute_channel_statistics(
    subset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    pixel_sum = torch.zeros(
        6,
        dtype=torch.float64,
    )

    pixel_sq_sum = torch.zeros(
        6,
        dtype=torch.float64,
    )

    pixel_count = 0

    class_counts = torch.zeros(
        2,
        dtype=torch.long,
    )

    for images, masks in loader:

        images = images.to(torch.float64)

        pixel_sum += images.sum(
            dim=(0, 2, 3)
        )

        pixel_sq_sum += (
            images ** 2
        ).sum(
            dim=(0, 2, 3)
        )

        pixel_count += (
            images.shape[0]
            * images.shape[2]
            * images.shape[3]
        )

        class_counts += torch.bincount(
            masks.reshape(-1),
            minlength=2,
        )

    mean = pixel_sum / pixel_count

    variance = (
        pixel_sq_sum / pixel_count
    ) - mean.square()

    std = (
        variance
        .clamp_min(1e-12)
        .sqrt()
    )

    return (
        mean.float(),
        std.float(),
        class_counts,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(train_class_counts, cap=4.0):

    negative_pixels, flood_pixels = (
        train_class_counts.tolist()
    )

    flood_weight = min(
        math.sqrt(
            negative_pixels
            / max(flood_pixels, 1)
        ),
        cap,
    )

    class_weights = torch.tensor(
        [1.0, flood_weight],
        dtype=torch.float32,
    )

    return class_weights, flood_weight


# ============================================================
# TRANSFORM DATASET
# ============================================================

class FloodTransformDataset(Dataset):

    def __init__(
        self,
        subset,
        mean,
        std,
        augment=False,
    ):

        self.subset = subset

        self.mean = mean[:, None, None]

        self.std = (
            std[:, None, None]
            .clamp_min(1e-6)
        )

        self.augment = augment

    def __len__(self):

        return len(self.subset)

    def __getitem__(self, index):

        image, mask = self.subset[index]

        # ----------------------------------------------------
        # Spatial augmentation
        # ----------------------------------------------------

        if self.augment:

            if torch.rand(()) < 0.5:

                image, mask = (
                    image.flip(-1),
                    mask.flip(-1),
                )

            if torch.rand(()) < 0.5:

                image, mask = (
                    image.flip(-2),
                    mask.flip(-2),
                )

            turns = int(
                torch.randint(
                    0,
                    4,
                    (),
                ).item()
            )

            if turns:

                image, mask = (
                    torch.rot90(
                        image,
                        turns,
                        dims=(-2, -1),
                    ),
                    torch.rot90(
                        mask,
                        turns,
                        dims=(-2, -1),
                    ),
                )

        # ----------------------------------------------------
        # Train-only normalization
        # ----------------------------------------------------

        image = (
            image - self.mean
        ) / self.std

        return image, mask


# ============================================================
# EXACT DISTRIBUTED EVALUATION SAMPLER
# ============================================================

class DistributedEvalSampler(Sampler):

    """
    Distributed sampler for validation/test that does not
    pad or duplicate samples.

    Every validation/test sample is evaluated exactly once
    across all DDP processes.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
    ):

        if num_replicas is None:
            num_replicas = dist.get_world_size()

        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.indices = list(
            range(
                rank,
                len(dataset),
                num_replicas,
            )
        )

    def __iter__(self):

        return iter(self.indices)

    def __len__(self):

        return len(self.indices)


# ============================================================
# CREATE COMPLETE DATA PIPELINE
# ============================================================

def create_data_pipeline(
    train_dataset,
    val_dataset,
    test_dataset,
    channel_mean,
    channel_std,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=SEED,
):
    """
    Create transformed datasets, DDP samplers, and DataLoaders.

    This function must be called AFTER DDP initialization.
    """

    transformer_train_dataset = FloodTransformDataset(
        train_dataset,
        channel_mean,
        channel_std,
        augment=True,
    )

    transformer_val_dataset = FloodTransformDataset(
        val_dataset,
        channel_mean,
        channel_std,
        augment=False,
    )

    transformer_test_dataset = FloodTransformDataset(
        test_dataset,
        channel_mean,
        channel_std,
        augment=False,
    )

    ddp_active = (
        dist.is_available()
        and dist.is_initialized()
    )

    if ddp_active:

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        train_sampler = DistributedSampler(
            transformer_train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        # IMPORTANT:
        # Do not use DistributedSampler for evaluation because
        # it can pad the dataset and duplicate samples.
        val_sampler = DistributedEvalSampler(
            transformer_val_dataset,
            num_replicas=world_size,
            rank=rank,
        )

        test_sampler = DistributedEvalSampler(
            transformer_test_dataset,
            num_replicas=world_size,
            rank=rank,
        )

    else:

        train_sampler = None
        val_sampler = None
        test_sampler = None

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:

        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    train_loader_generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(
        transformer_train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        generator=train_loader_generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        transformer_val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        transformer_test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    return (
        transformer_train_dataset,
        transformer_val_dataset,
        transformer_test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        val_sampler,
        test_sampler,
    )


# ============================================================
# COMPLETE DATA PREPARATION
# ============================================================

def prepare_data(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=SEED,
    flood_weight_cap=FLOOD_WEIGHT_CAP,
):
    """
    End-to-end data preparation:

        1. Build the raw Sen2GF3Floods dataset.
        2. Split it into train/val/test (seeded).
        3. Compute train-only channel mean/std and class counts.
        4. Derive class weights from the class counts, capped at
           `flood_weight_cap` (see config.FLOOD_WEIGHT_CAP for the
           rationale behind the current experiment's value).
        5. Build normalized/augmented datasets, DDP-aware
           samplers, and DataLoaders.

    NOTE: this must be called AFTER DDP has been initialized
    (see training.setup_ddp) so that the DistributedSampler /
    DistributedEvalSampler branches pick up the correct
    world_size and rank.
    """

    dataset = create_dataset()

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_dataset_splits(
        dataset,
        seed=seed,
    )

    (
        channel_mean,
        channel_std,
        train_class_counts,
    ) = compute_channel_statistics(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_weights, flood_weight = (
        compute_class_weights(
            train_class_counts,
            cap=flood_weight_cap,
        )
    )

    (
        transformer_train_dataset,
        transformer_val_dataset,
        transformer_test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = create_data_pipeline(
        train_dataset,
        val_dataset,
        test_dataset,
        channel_mean,
        channel_std,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    return {
        "dataset": dataset,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "transformer_train_dataset": transformer_train_dataset,
        "transformer_val_dataset": transformer_val_dataset,
        "transformer_test_dataset": transformer_test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "test_sampler": test_sampler,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "train_class_counts": train_class_counts,
        "class_weights": class_weights,
        "flood_weight": flood_weight,
    }
