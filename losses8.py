import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Foreground Tversky loss.

    The Tversky index generalizes Dice by weighting false positives
    and false negatives independently:

        TI = TP / (TP + alpha * FP + beta * FN)

    alpha == beta == 0.5  ->  identical to standard Dice loss.
    alpha  >  beta        ->  false positives cost more than false
                               negatives, which pushes the model
                               toward higher PRECISION (fewer, more
                               confident foreground predictions).
    alpha  <  beta        ->  the opposite: favors higher RECALL.
    """

    def __init__(
        self,
        alpha=0.7,
        beta=0.3,
        smooth=1.0,
    ):
        super().__init__()

        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError(
                "alpha and beta must both be in [0, 1]. "
                f"Got alpha={alpha}, beta={beta}."
            )

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):

        flood_probability = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        flood_target = (
            targets == 1
        ).float()

        # ----------------------------------------------------
        # Soft TP / FP / FN, per-sample
        # ----------------------------------------------------

        true_positive = (
            flood_probability * flood_target
        ).sum(dim=(1, 2))

        false_positive = (
            flood_probability * (1.0 - flood_target)
        ).sum(dim=(1, 2))

        false_negative = (
            (1.0 - flood_probability) * flood_target
        ).sum(dim=(1, 2))

        tversky_index = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return 1.0 - tversky_index.mean()


class FloodCompositeLoss(nn.Module):
    """
    Composite loss for flood segmentation.

    Loss = weighted cross-entropy + region_weight * region_loss

    region_loss is selected via `region_mode`:
      - "dice"    : standard Dice loss (Tversky with alpha=beta=0.5)
      - "tversky" : Tversky loss with tunable alpha/beta (default),
                    configured here to penalize false positives more
                    than false negatives so the model favors precision.
    """

    def __init__(
        self,
        class_weights,
        region_weight=None,
        region_mode="tversky",
        tversky_alpha=0.7,
        tversky_beta=0.3,
        smooth=1.0,
        dice_weight=0.5,
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            class_weights.float(),
        )

        # Backward-compatible: if the caller still passes
        # `dice_weight`, use it unless `region_weight` was set
        # explicitly.
        self.region_weight = (
            dice_weight
            if region_weight is None
            else region_weight
        )

        self.region_mode = region_mode

        if region_mode == "dice":
            self.region_loss = TverskyLoss(
                alpha=0.5,
                beta=0.5,
                smooth=smooth,
            )

        elif region_mode == "tversky":
            self.region_loss = TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                smooth=smooth,
            )

        else:
            raise ValueError(
                f"Unknown region_mode: {region_mode!r}. "
                "Expected 'dice' or 'tversky'."
            )

    def forward(self, logits, targets):
        # ----------------------------------------------------
        # Weighted cross-entropy
        # ----------------------------------------------------

        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
        )

        # ----------------------------------------------------
        # Region loss (Dice or precision-favoring Tversky)
        # ----------------------------------------------------

        region_loss = self.region_loss(
            logits,
            targets,
        )

        # ----------------------------------------------------
        # Composite loss
        # ----------------------------------------------------

        return (
            cross_entropy
            + self.region_weight * region_loss
        )
