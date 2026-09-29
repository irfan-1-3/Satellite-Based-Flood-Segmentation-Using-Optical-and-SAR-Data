# ============================================================
# DDP / AMP TRAINING UTILITIES
# ============================================================

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm


# ============================================================
# DDP SETUP
# ============================================================

def setup_ddp():
    """
    Initialize real multi-GPU DDP when launched with `torchrun`
    (RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
    are all present in the environment). Otherwise fall back to
    a single-process, single-GPU run on GPU 0.

    Returns:
        local_rank, rank, world_size, device
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    num_visible_gpus = torch.cuda.device_count()

    for gpu_id in range(num_visible_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    launched_with_torchrun = all(
        key in os.environ
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
    )

    if launched_with_torchrun:

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )

        if rank == 0:
            print(f"\nDDP initialized: world_size={world_size}")
            print("Real multi-GPU training is ACTIVE.")

    else:

        local_rank = 0
        rank = 0
        world_size = 1

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        print(
            "\nRunning as a plain single process "
            "(no torchrun launcher detected)."
        )
        print(
            f"{num_visible_gpus} GPU(s) are visible, but "
            "dist.init_process_group() is NOT called here, so "
            "training will use only GPU 0. Launch with torchrun "
            "to use every visible GPU via real DDP."
        )

    return local_rank, rank, world_size, device


# ============================================================
# DDP HELPERS
# ============================================================

def is_ddp():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_ddp() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if is_ddp() else 0


def get_world_size():
    return dist.get_world_size() if is_ddp() else 1


def get_model_for_save(model):
    """
    Return the underlying model when using DDP.
    Avoids saving 'module.' prefixes.
    """

    if isinstance(model, DDP):
        return model.module

    return model


# ============================================================
# PER-RANK RNG STATE
# ============================================================

def get_rng_state():

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()

    return state


def restore_rng_state(state):
    if state is None:
        return

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch_state = torch.as_tensor(
            state["torch"],
            dtype=torch.uint8,
            device="cpu",
        )
        torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and "cuda" in state:
        cuda_state = torch.as_tensor(
            state["cuda"],
            dtype=torch.uint8,
            device="cpu",
        )
        torch.cuda.set_rng_state(cuda_state)


def collect_rng_states():

    local_state = get_rng_state()

    if not is_ddp():
        return [local_state]

    gathered_states = [None for _ in range(get_world_size())]

    dist.all_gather_object(gathered_states, local_state)

    return gathered_states


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_miou,
    best_metrics,
    history,
    amp_enabled,
    rng_states=None,
):

    model_to_save = get_model_for_save(model)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "best_metrics": best_metrics,
        "history": history,
        "amp_enabled": amp_enabled,
        # Store RNG state for every rank.
        "rng_states": rng_states,
    }

    torch.save(checkpoint, path)


# ============================================================
# METRICS FROM CONFUSION COUNTS (memory-efficient, DDP-safe)
# ============================================================

def calculate_metrics_from_counts(tp, tn, fp, fn):

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0

    iou_flood = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    miou = (iou_background + iou_flood) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miou": miou,
    }


# ============================================================
# MEMORY-EFFICIENT VALIDATION / TEST — DDP AWARE
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    use_amp=True,
):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    amp_enabled = use_amp and device.type == "cuda"

    for images, masks in loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_batches += 1

        predictions = torch.argmax(outputs, dim=1)

        total_tp += ((predictions == 1) & (masks == 1)).sum().item()
        total_tn += ((predictions == 0) & (masks == 0)).sum().item()
        total_fp += ((predictions == 1) & (masks == 0)).sum().item()
        total_fn += ((predictions == 0) & (masks == 1)).sum().item()

    # --------------------------------------------------------
    # DDP — aggregate statistics across all GPUs
    # --------------------------------------------------------

    if dist.is_available() and dist.is_initialized():

        stats = torch.tensor(
            [
                total_loss,
                total_batches,
                total_tp,
                total_tn,
                total_fp,
                total_fn,
            ],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_loss = stats[0].item()
        total_batches = int(stats[1].item())
        total_tp = int(stats[2].item())
        total_tn = int(stats[3].item())
        total_fp = int(stats[4].item())
        total_fn = int(stats[5].item())

    val_loss = total_loss / total_batches if total_batches > 0 else 0.0

    metrics = calculate_metrics_from_counts(
        tp=total_tp,
        tn=total_tn,
        fp=total_fp,
        fn=total_fn,
    )

    return metrics, val_loss


# ============================================================
# RESUMABLE DDP + AMP TRAINING
# ============================================================

def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    num_epochs,
    device,
    checkpoint_dir,
    criterion,
    optimizer_factory,
    scheduler_factory,
    scheduler_step_mode,
    use_amp=True,
    max_grad_norm=1.0,
    resume_training=True,
    checkpoint_every=5,
):
    """
    Train `model` with AMP, optional DDP, and resumable
    checkpointing.

    `resume_training` / `checkpoint_every` are explicit
    parameters (not globals) so this function is safe to call
    from any entry point / script.
    """

    # ========================================================
    # PROCESS INFORMATION
    # ========================================================

    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():

        print("=" * 60)
        print(f"TRAINING: {model_name}")
        print("=" * 60)
        print(f"DDP enabled : {is_ddp()}")
        print(f"World size  : {world_size}")
        print(f"Device      : {device}")
        print("=" * 60)

    # ========================================================
    # AMP
    # ========================================================

    amp_enabled = use_amp and device.type == "cuda"

    if is_main_process():

        print(f"AMP enabled: {amp_enabled}")

        if amp_enabled:
            print("AMP dtype: float16")

    # ========================================================
    # MODEL -> DEVICE
    # ========================================================

    model = model.to(device)

    # ========================================================
    # LOSS
    # ========================================================

    if criterion is None:
        raise ValueError("criterion must be provided.")

    # ========================================================
    # OPTIMIZER
    #
    # IMPORTANT: create optimizer BEFORE DDP wrapping.
    # ========================================================

    if optimizer_factory is None:
        raise ValueError("optimizer_factory must be provided.")

    optimizer = optimizer_factory(model)

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    if scheduler_factory is None:
        raise ValueError("scheduler_factory must be provided.")

    scheduler = scheduler_factory(optimizer)

    # ========================================================
    # AMP GRADIENT SCALER
    # ========================================================

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ========================================================
    # DDP MODEL WRAPPING
    # ========================================================

    if is_ddp():

        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    # ========================================================
    # CHECKPOINT PATHS
    # ========================================================

    model_checkpoint_dir = os.path.join(
        checkpoint_dir,
        model_name.replace(" ", "_").replace("+", "plus"),
    )

    os.makedirs(model_checkpoint_dir, exist_ok=True)

    latest_checkpoint = os.path.join(
        model_checkpoint_dir, "checkpoint_latest.pth"
    )

    best_checkpoint = os.path.join(
        model_checkpoint_dir, "checkpoint_best.pth"
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    start_epoch = 0
    best_miou = -1.0

    best_metrics = {
        "loss": float("inf"),
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "miou": 0.0,
    }

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_miou": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "miou": [],
        "lr": [],
    }

    # ========================================================
    # RESUME FROM CHECKPOINT
    # ========================================================

    if resume_training and os.path.exists(latest_checkpoint):

        if is_main_process():
            print("\nCheckpoint found:")
            print(latest_checkpoint)

        # All ranks load the same checkpoint.
        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False,
        )

        get_model_for_save(model).load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if amp_enabled and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", -1.0)
        best_metrics = checkpoint.get("best_metrics", best_metrics)
        history = checkpoint.get("history", history)

        # Backward compatibility: older checkpoints (e.g. from the
        # previous Dice-loss run) won't have train_accuracy/train_miou
        # in their history dict. Backfill with None placeholders so
        # resuming doesn't crash on history["train_accuracy"].append(...).
        for key in ("train_accuracy", "train_miou"):
            if key not in history:
                history[key] = [None] * len(history["epoch"])

        # ----------------------------------------------------
        # Restore per-rank RNG
        # ----------------------------------------------------

        rng_states = checkpoint.get("rng_states", None)

        if rng_states is not None:

            if rank < len(rng_states):
                restore_rng_state(rng_states[rank])

        # Backward compatibility with older checkpoints.
        elif "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

        if is_main_process():
            print(f"Resuming from epoch {start_epoch + 1}/{num_epochs}")
            print(f"Best validation mIoU so far: {best_miou:.6f}")

    else:

        if is_main_process():
            print("\nNo checkpoint found.")
            print("Starting from epoch 1.")

    # ========================================================
    # SYNCHRONIZE ALL PROCESSES
    # ========================================================

    if is_ddp():
        dist.barrier()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(start_epoch, num_epochs):

        # ----------------------------------------------------
        # Distributed epoch seed
        # ----------------------------------------------------

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()

        running_loss = 0.0
        local_batch_count = 0

        # Training-set confusion-matrix accumulators, aggregated the
        # same DDP-safe way as validation, so we get train_accuracy
        # and train_miou alongside train_loss with no extra forward
        # pass (reuses the logits already computed for the loss).
        local_train_tp = 0
        local_train_tn = 0
        local_train_fp = 0
        local_train_fn = 0

        if is_main_process():
            progress_bar = tqdm(
                train_loader,
                desc=f"{model_name} Epoch {epoch + 1}/{num_epochs}",
            )
        else:
            progress_bar = train_loader

        # ====================================================
        # TRAINING
        # ====================================================

        for images, masks in progress_bar:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):

                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            local_batch_count += 1

            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)

                local_train_tp += ((predictions == 1) & (masks == 1)).sum().item()
                local_train_tn += ((predictions == 0) & (masks == 0)).sum().item()
                local_train_fp += ((predictions == 1) & (masks == 0)).sum().item()
                local_train_fn += ((predictions == 0) & (masks == 1)).sum().item()

            if is_main_process():
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ====================================================
        # GLOBAL TRAINING LOSS + TRAINING METRICS
        # ====================================================

        train_stats_tensor = torch.tensor(
            [
                running_loss,
                local_batch_count,
                local_train_tp,
                local_train_tn,
                local_train_fp,
                local_train_fn,
            ],
            dtype=torch.float64,
            device=device,
        )

        if is_ddp():
            dist.all_reduce(train_stats_tensor, op=dist.ReduceOp.SUM)

        global_running_loss = train_stats_tensor[0].item()
        global_batch_count = train_stats_tensor[1].item()
        global_train_tp = int(train_stats_tensor[2].item())
        global_train_tn = int(train_stats_tensor[3].item())
        global_train_fp = int(train_stats_tensor[4].item())
        global_train_fn = int(train_stats_tensor[5].item())

        train_loss = (
            global_running_loss / global_batch_count
            if global_batch_count > 0
            else 0.0
        )

        train_metrics = calculate_metrics_from_counts(
            tp=global_train_tp,
            tn=global_train_tn,
            fp=global_train_fp,
            fn=global_train_fn,
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_metrics, val_loss = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=amp_enabled,
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        if scheduler_step_mode == "val_loss":
            scheduler.step(val_loss)
        elif scheduler_step_mode == "epoch":
            scheduler.step()
        else:
            raise ValueError(
                "scheduler_step_mode must be 'val_loss' or 'epoch'"
            )

        current_lr = max(group["lr"] for group in optimizer.param_groups)

        # ====================================================
        # STORE METRICS
        # ====================================================

        if is_main_process():

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_miou"].append(train_metrics["miou"])
            history["val_loss"].append(val_loss)
            history["accuracy"].append(val_metrics["accuracy"])
            history["precision"].append(val_metrics["precision"])
            history["recall"].append(val_metrics["recall"])
            history["f1"].append(val_metrics["f1"])
            history["miou"].append(val_metrics["miou"])
            history["lr"].append(current_lr)

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        if is_main_process():

            print("\n" + "-" * 60)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss     : {train_loss:.6f}")
            print(f"Train Accuracy : {train_metrics['accuracy']:.6f}")
            print(f"Train mIoU     : {train_metrics['miou']:.6f}")
            print(f"Val Loss   : {val_loss:.6f}")
            print(f"Accuracy   : {val_metrics['accuracy']:.6f}")
            print(f"Precision  : {val_metrics['precision']:.6f}")
            print(f"Recall     : {val_metrics['recall']:.6f}")
            print(f"F1         : {val_metrics['f1']:.6f}")
            print(f"mIoU       : {val_metrics['miou']:.6f}")
            print(f"Learning Rate : {current_lr:.2e}")

        # ====================================================
        # COLLECT PER-RANK RNG STATES
        # ====================================================

        if is_ddp():
            rng_states = collect_rng_states()
        else:
            rng_states = [get_rng_state()]

        # ====================================================
        # BEST MODEL
        # ====================================================

        if is_main_process():

            if val_metrics["miou"] > best_miou:

                best_miou = val_metrics["miou"]

                best_metrics = {
                    "loss": val_loss,
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "miou": val_metrics["miou"],
                }

                save_checkpoint(
                    path=best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_miou=best_miou,
                    best_metrics=best_metrics,
                    history=history,
                    amp_enabled=amp_enabled,
                    rng_states=rng_states,
                )

                print("\n*** NEW BEST MODEL SAVED ***")

        # ====================================================
        # LATEST + PERIODIC CHECKPOINTS
        # ====================================================

        if is_main_process():

            save_checkpoint(
                path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_miou=best_miou,
                best_metrics=best_metrics,
                history=history,
                amp_enabled=amp_enabled,
                rng_states=rng_states,
            )

            if (epoch + 1) % checkpoint_every == 0:

                periodic_checkpoint = os.path.join(
                    model_checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1:03d}.pth",
                )

                save_checkpoint(
                    path=periodic_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_miou=best_miou,
                    best_metrics=best_metrics,
                    history=history,
                    amp_enabled=amp_enabled,
                    rng_states=rng_states,
                )

                print(f"Periodic checkpoint saved: {periodic_checkpoint}")

        # ====================================================
        # SYNCHRONIZE BEFORE NEXT EPOCH
        # ====================================================

        if is_ddp():
            dist.barrier()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if is_main_process():

        if len(history["miou"]) > 0:
            best_epoch_index = int(np.argmax(history["miou"]))
            best_epoch = history["epoch"][best_epoch_index]
        else:
            best_epoch = None

        print("\n" + "=" * 60)
        print(f"{model_name} TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Epoch       : {best_epoch}")
        print(f"Validation Loss  : {best_metrics['loss']:.6f}")
        print(f"Accuracy         : {best_metrics['accuracy']:.6f}")
        print(f"Precision        : {best_metrics['precision']:.6f}")
        print(f"Recall           : {best_metrics['recall']:.6f}")
        print(f"F1               : {best_metrics['f1']:.6f}")
        print(f"mIoU             : {best_metrics['miou']:.6f}")
        print(f"\nBest model saved at:\n{best_checkpoint}")
        print("=" * 60)

    return model, history, best_metrics
