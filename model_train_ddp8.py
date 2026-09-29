import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist

from config import (
    SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_ID,
    NUM_CLASSES,
    DECODER_DIM,
    CHECKPOINT_DIR,
    REGION_LOSS_MODE,
    REGION_LOSS_WEIGHT,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
)

from dataset import prepare_data

from model import (
    DualStreamCrossModalSegFormer,
)

from losses import (
    FloodCompositeLoss,
)

from training import (
    setup_ddp,
    train_model,
    evaluate_model,
    is_ddp,
    is_main_process,
    get_model_for_save,
)


# ============================================================
# SEED / PERFORMANCE
# ============================================================

def set_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================
# DISCRIMINATIVE LEARNING RATES
# ============================================================
# Pretrained transformer encoders need a much smaller LR than the new
# fusion and decoder layers. A single global LR can erase useful
# pretrained ImageNet features before they adapt to flood imagery.

TRANSFORMER_ENCODER_LR = 1e-5
TRANSFORMER_HEAD_LR = 1e-4
TRANSFORMER_WEIGHT_DECAY = 1e-2
TRANSFORMER_WARMUP_EPOCHS = 5
TRANSFORMER_MIN_LR = 1e-6
TRANSFORMER_MAX_GRAD_NORM = 1.0


# ============================================================
# OPTIMIZER (discriminative LR: encoders vs. fusion/decoder head)
# ============================================================

def optimizer_factory(model):

    encoder_parameters = (
        list(model.optical_encoder.parameters())
        + list(model.sar_encoder.parameters())
    )

    encoder_parameter_ids = {
        id(parameter)
        for parameter in encoder_parameters
    }

    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in encoder_parameter_ids
    ]

    return optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": TRANSFORMER_ENCODER_LR,
            },
            {
                "params": head_parameters,
                "lr": TRANSFORMER_HEAD_LR,
            },
        ],
        weight_decay=TRANSFORMER_WEIGHT_DECAY,
    )


# ============================================================
# LR SCHEDULER (linear warmup -> cosine annealing)
# ============================================================

def scheduler_factory(optimizer):

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=TRANSFORMER_WARMUP_EPOCHS,
    )

    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - TRANSFORMER_WARMUP_EPOCHS,
        eta_min=TRANSFORMER_MIN_LR,
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup, cosine],
        milestones=[TRANSFORMER_WARMUP_EPOCHS],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # IMPORTANT:
    # DDP MUST BE INITIALIZED FIRST.
    # --------------------------------------------------------

    local_rank, rank, world_size, device = setup_ddp()

    set_seed(SEED + rank)

    if is_main_process():

        print("=" * 60)
        print("DUAL-STREAM CROSS-MODAL SEGFORMER")
        print("=" * 60)

        print(
            f"World size : {world_size}"
        )

        print(
            f"Device     : {device}"
        )

        print(
            f"Batch/GPU  : {BATCH_SIZE}"
        )

        print(
            f"Effective batch size : "
            f"{BATCH_SIZE * world_size}"
        )

        print("=" * 60)

    # ========================================================
    # DATA
    # ========================================================

    data = prepare_data(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    class_weights = data["class_weights"]

    # ========================================================
    # MODEL
    # ========================================================

    model = DualStreamCrossModalSegFormer(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        decoder_dim=DECODER_DIM,
        rank=rank,
        is_ddp=is_ddp(),
    )

    model = model.to(device)

    # ========================================================
    # LOSS
    # ========================================================

    criterion = FloodCompositeLoss(
        class_weights=class_weights.to(device),
        region_weight=REGION_LOSS_WEIGHT,
        region_mode=REGION_LOSS_MODE,
        # alpha > beta: penalize false positives more than false
        # negatives, directly targeting the precision gap
        tversky_alpha=TVERSKY_ALPHA,
        tversky_beta=TVERSKY_BETA,
    ).to(device)

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if is_main_process():

        sample_images, sample_masks = next(
            iter(train_loader)
        )

        sample_images = sample_images.to(
            device
        )

        sample_masks = sample_masks.to(
            device
        )

        with torch.no_grad():

            sample_logits = model(
                sample_images
            )

            sample_loss = criterion(
                sample_logits,
                sample_masks,
            )

        print("=" * 60)
        print("MODEL SANITY CHECK")
        print("=" * 60)

        print(
            "Input shape  :",
            tuple(sample_images.shape),
        )

        print(
            "Logit shape  :",
            tuple(sample_logits.shape),
        )

        print(
            "Mask shape   :",
            tuple(sample_masks.shape),
        )

        print(
            f"Initial loss : "
            f"{sample_loss.item():.6f}"
        )

        print("=" * 60)

    # ========================================================
    # SYNCHRONIZE BEFORE TRAINING
    # ========================================================

    if is_ddp():
        dist.barrier()

    # ========================================================
    # TRAIN
    # ========================================================

    trained_model, history, best_metrics = (
        train_model(
            model=model,
            model_name=(
                "DualStreamCrossModalSegFormer"
            ),
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            criterion=criterion,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_step_mode="epoch",
            use_amp=True,
            max_grad_norm=TRANSFORMER_MAX_GRAD_NORM,
            resume_training=True,
            checkpoint_every=5,
        )
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    best_checkpoint_path = (
        CHECKPOINT_DIR
        / "DualStreamCrossModalSegFormer"
        / "checkpoint_best.pth"
    )

    if not best_checkpoint_path.exists():

        raise FileNotFoundError(
            "Best checkpoint was not found:\n"
            f"{best_checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    get_model_for_save(
        trained_model
    ).load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # SYNCHRONIZE
    # ========================================================

    if is_ddp():
        dist.barrier()

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    test_metrics, test_loss = (
        evaluate_model(
            model=trained_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=True,
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    if is_main_process():

        print()
        print("=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)

        print(
            f"Test Loss  : "
            f"{test_loss:.6f}"
        )

        print(
            f"Accuracy   : "
            f"{test_metrics['accuracy']:.6f}"
        )

        print(
            f"Precision  : "
            f"{test_metrics['precision']:.6f}"
        )

        print(
            f"Recall     : "
            f"{test_metrics['recall']:.6f}"
        )

        print(
            f"F1         : "
            f"{test_metrics['f1']:.6f}"
        )

        print(
            f"mIoU       : "
            f"{test_metrics['miou']:.6f}"
        )

        print()
        print(
            "Best checkpoint:"
        )
        print(
            best_checkpoint_path
        )

        print("=" * 60)

    # ========================================================
    # FINAL SYNCHRONIZATION
    # ========================================================

    if is_ddp():
        dist.barrier()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
