# ============================================================
# DUAL-STREAM CROSS-MODAL SEGFORMER
# ============================================================

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerModel

from config import (
    MODEL_ID,
    NUM_CLASSES,
    DECODER_DIM,
    HIGH_RES_BRANCH_HIDDEN,
    HIGH_RES_BRANCH_OUT,
)


# ============================================================
# PRETRAINED SEGFORMER INPUT PROJECTION
# ============================================================

def get_first_patch_embedding(model):
    """
    Return SegFormer's first patch-embedding module
    across Transformers versions.
    """

    if hasattr(model, "stages"):
        return model.stages[0].patch_embeddings

    if hasattr(model, "encoder"):
        return model.encoder.patch_embeddings[0]

    raise AttributeError(
        "Could not find SegFormer's first patch-embedding module. "
        "Expected either `model.stages` or `model.encoder`."
    )


def replace_patch_projection(
    model,
    input_channels,
    init_mode,
):
    """
    Adapt a pretrained RGB patch projection to
    optical or SAR inputs.
    """

    patch_embedding = get_first_patch_embedding(model)

    old_projection = patch_embedding.proj

    new_projection = nn.Conv2d(
        input_channels,
        old_projection.out_channels,
        kernel_size=old_projection.kernel_size,
        stride=old_projection.stride,
        padding=old_projection.padding,
        bias=old_projection.bias is not None,
    )

    with torch.no_grad():

        rgb_mean = (
            old_projection.weight
            .mean(dim=1, keepdim=True)
        )

        if init_mode == "optical":

            # Preserve pretrained RGB weights.
            new_projection.weight[:, :3].copy_(
                old_projection.weight
            )

            # Initialize B8 / NIR from RGB mean.
            new_projection.weight[:, 3:4].copy_(
                rgb_mean
            )

        elif init_mode == "sar":

            # Initialize HH/HV from RGB mean.
            new_projection.weight.copy_(
                rgb_mean.expand(
                    -1,
                    input_channels,
                    -1,
                    -1,
                )
            )

        else:

            raise ValueError(
                f"Unknown initialization mode: {init_mode}"
            )

        if old_projection.bias is not None:

            new_projection.bias.copy_(
                old_projection.bias
            )

    patch_embedding.proj = new_projection


# ============================================================
# HIGH-RESOLUTION DECODER BRANCH
# ============================================================

class HighResolutionBranch(nn.Module):
    """
    Shallow, full-resolution feature extractor over the raw
    6-channel input (4 optical + 2 SAR), run in parallel to the
    two SegFormer encoders.

    This is NOT a replacement for the encoders and does not
    itself do any semantic reasoning about "is this flood" -- its
    only job is to preserve fine spatial detail (edges, thin/small
    structures) that the encoders' patch embeddings destroy via
    strided downsampling, and hand it to the decoder as an
    additional, full-resolution input.

    Never downsamples: every conv uses stride 1 with same-padding,
    so the output stays at the input's native 256x256 resolution.
    """

    def __init__(
        self,
        in_channels=6,
        hidden_dim=HIGH_RES_BRANCH_HIDDEN,
        out_dim=HIGH_RES_BRANCH_OUT,
    ):

        super().__init__()

        self.branch = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                out_dim,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, images):
        return self.branch(images)


# ============================================================
# FUSION STAGE
# ============================================================

class FusionStage(nn.Module):
    """
    Fuse equal-scale optical and SAR features.

    Shallow stages use gated fusion.
    Deep stages use bidirectional cross-attention.
    """

    def __init__(
        self,
        channels,
        heads,
        use_cross_attention,
    ):

        super().__init__()

        self.use_cross_attention = (
            use_cross_attention
        )

        if use_cross_attention:

            self.optical_norm = nn.LayerNorm(
                channels
            )

            self.sar_norm = nn.LayerNorm(
                channels
            )

            self.optical_to_sar = (
                nn.MultiheadAttention(
                    channels,
                    heads,
                    batch_first=True,
                )
            )

            self.sar_to_optical = (
                nn.MultiheadAttention(
                    channels,
                    heads,
                    batch_first=True,
                )
            )

        else:

            self.gate = nn.Sequential(

                nn.Conv2d(
                    channels * 2,
                    channels,
                    kernel_size=1,
                    bias=False,
                ),

                nn.GroupNorm(
                    8,
                    channels,
                ),

                nn.GELU(),

                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=1,
                ),

                nn.Sigmoid(),
            )

        self.merge = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                channels,
            ),

            nn.GELU(),
        )

    def forward(
        self,
        optical,
        sar,
    ):

        if self.use_cross_attention:

            batch, channels, height, width = (
                optical.shape
            )

            optical_tokens = (
                optical
                .flatten(2)
                .transpose(1, 2)
            )

            sar_tokens = (
                sar
                .flatten(2)
                .transpose(1, 2)
            )

            # Optical queries SAR.
            optical_update, _ = (
                self.optical_to_sar(
                    self.optical_norm(
                        optical_tokens
                    ),
                    self.sar_norm(
                        sar_tokens
                    ),
                    self.sar_norm(
                        sar_tokens
                    ),
                    need_weights=False,
                )
            )

            # SAR queries optical.
            sar_update, _ = (
                self.sar_to_optical(
                    self.sar_norm(
                        sar_tokens
                    ),
                    self.optical_norm(
                        optical_tokens
                    ),
                    self.optical_norm(
                        optical_tokens
                    ),
                    need_weights=False,
                )
            )

            optical = (
                optical_tokens
                + optical_update
            ).transpose(
                1,
                2,
            ).reshape(
                batch,
                channels,
                height,
                width,
            )

            sar = (
                sar_tokens
                + sar_update
            ).transpose(
                1,
                2,
            ).reshape(
                batch,
                channels,
                height,
                width,
            )

        else:

            weights = self.gate(
                torch.cat(
                    [optical, sar],
                    dim=1,
                )
            )

            optical = optical * (
                1.0
                + weights[
                    :,
                    :optical.shape[1],
                ]
            )

            sar = sar * (
                1.0
                + weights[
                    :,
                    optical.shape[1]:,
                ]
            )

        return self.merge(
            torch.cat(
                [optical, sar],
                dim=1,
            )
        )


# ============================================================
# DUAL-STREAM CROSS-MODAL SEGFORMER
# ============================================================

class DualStreamCrossModalSegFormer(nn.Module):
    """
    Transformer flood segmenter for:

        Optical: 4 channels [B4, B3, B2, B8]
        SAR:     2 channels [HH, HV]
    """

    def __init__(
        self,
        model_id=MODEL_ID,
        decoder_dim=DECODER_DIM,
        num_classes=NUM_CLASSES,
        rank=0,
        is_ddp=False,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load pretrained MiT-B2
        #
        # Rank 0 downloads first.
        # Other ranks wait and then load locally.
        # ----------------------------------------------------

        if rank == 0:

            SegformerModel.from_pretrained(
                model_id
            )

        if is_ddp:

            dist.barrier()

        pretrained_encoder = (
            SegformerModel.from_pretrained(
                model_id,
                local_files_only=True,
            )
        )

        # ----------------------------------------------------
        # Keep two independent pretrained streams
        # ----------------------------------------------------

        self.optical_encoder = copy.deepcopy(
            pretrained_encoder
        )

        self.sar_encoder = copy.deepcopy(
            pretrained_encoder
        )

        # ----------------------------------------------------
        # Adapt first projections
        # ----------------------------------------------------

        replace_patch_projection(
            self.optical_encoder,
            input_channels=4,
            init_mode="optical",
        )

        replace_patch_projection(
            self.sar_encoder,
            input_channels=2,
            init_mode="sar",
        )

        # ----------------------------------------------------
        # Feature dimensions
        # ----------------------------------------------------

        hidden_sizes = (
            self.optical_encoder
            .config
            .hidden_sizes
        )

        attention_heads = [
            4,
            4,
            8,
            8,
        ]

        # ----------------------------------------------------
        # Multi-scale fusion
        # ----------------------------------------------------

        self.fusion_stages = nn.ModuleList(
            [
                FusionStage(
                    channels,
                    heads,
                    use_cross_attention=(
                        stage >= 2
                    ),
                )

                for stage, (
                    channels,
                    heads,
                ) in enumerate(
                    zip(
                        hidden_sizes,
                        attention_heads,
                    )
                )
            ]
        )

        # ----------------------------------------------------
        # High-resolution decoder branch (raw 6-channel input,
        # kept at native 256x256 resolution throughout)
        # ----------------------------------------------------

        self.high_res_branch = HighResolutionBranch(
            in_channels=6,
            hidden_dim=HIGH_RES_BRANCH_HIDDEN,
            out_dim=HIGH_RES_BRANCH_OUT,
        )

        # ----------------------------------------------------
        # Decoder projections
        # ----------------------------------------------------

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    decoder_dim,
                    kernel_size=1,
                )

                for channels in hidden_sizes
            ]
        )

        # ----------------------------------------------------
        # Decoder
        #
        # Input channels = one decoder_dim block per encoder
        # scale (4), plus the high-resolution branch's output
        # channels, since it's now concatenated in alongside
        # the four upsampled encoder scales.
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Conv2d(
                decoder_dim * len(hidden_sizes)
                + HIGH_RES_BRANCH_OUT,
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                32,
                decoder_dim,
            ),

            nn.GELU(),

            nn.Dropout2d(
                0.1
            ),

            nn.Conv2d(
                decoder_dim,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, images):

        # ----------------------------------------------------
        # Split modalities
        # ----------------------------------------------------

        optical = images[:, :4]
        # B4, B3, B2, B8

        sar = images[:, 4:]
        # HH, HV

        # ----------------------------------------------------
        # Optical encoder
        # ----------------------------------------------------

        optical_features = (
            self.optical_encoder(
                pixel_values=optical,
                output_hidden_states=True,
                return_dict=True,
            )
            .hidden_states
        )

        # ----------------------------------------------------
        # SAR encoder
        # ----------------------------------------------------

        sar_features = (
            self.sar_encoder(
                pixel_values=sar,
                output_hidden_states=True,
                return_dict=True,
            )
            .hidden_states
        )

        # ----------------------------------------------------
        # Multi-scale fusion
        # ----------------------------------------------------

        fused = [
            fusion(
                optical_feature,
                sar_feature,
            )

            for fusion,
            optical_feature,
            sar_feature

            in zip(
                self.fusion_stages,
                optical_features,
                sar_features,
            )
        ]

        # ----------------------------------------------------
        # High-resolution branch (native 256x256, no pooling)
        # ----------------------------------------------------

        high_res_features = self.high_res_branch(images)

        # ----------------------------------------------------
        # Decoder -- now decides at FULL input resolution
        #
        # Previously all four scales were upsampled only to
        # stage-0 resolution (64x64), decided there, and the
        # 2-class logits were bilinearly upsampled to 256x256
        # as a final, purely cosmetic step -- which cannot
        # recover crisp boundaries no matter how good the 64x64
        # features were. Upsampling every scale directly to the
        # true input resolution and adding the native-resolution
        # high-res branch means the decoder's actual per-pixel
        # decision is made with real full-resolution information.
        # ----------------------------------------------------

        target_size = images.shape[-2:]

        decoded = [
            F.interpolate(
                projection(feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            for projection,
            feature

            in zip(
                self.projections,
                fused,
            )
        ]

        decoded.append(high_res_features)

        logits = self.decoder(
            torch.cat(
                decoded,
                dim=1,
            )
        )

        # ----------------------------------------------------
        # No further upsampling needed: logits are already at
        # the input's native resolution.
        # ----------------------------------------------------

        return logits
