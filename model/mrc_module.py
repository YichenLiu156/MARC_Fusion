"""
MRF modules for MARCFusion.

This file implements two independent material-aware reliability-consistency
fusion modules:

1. LowFrequencyMRF:
   Material-guided consistency fusion for low-frequency features.

2. HighFrequencyMRF:
   Material-guided reliability selection for high-frequency features.

The modules are designed to be placed after frequency-branch decoding and
before OutputLayers / IDWT.

Expected inputs:
    low_ir, low_vis:
        [B, C, H_l, W_l]

    high_ir, high_vis:
        [B, 3C, H_h, W_h]

    z_m:
        [B, C_m, H, W] or any spatial size. It will be resized internally.

Outputs:
    low_fused:
        [B, C, H_l, W_l]

    high_fused:
        [B, 3C, H_h, W_h]

No dependency on old FrequenceDomainInteractionBlock or output_attn.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    """Small Conv-BN-PReLU block used inside MRF modules."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_bn: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not use_bn,
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_act:
            layers.append(nn.PReLU(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def resize_like(
    x: torch.Tensor,
    ref: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize x to the spatial size of ref if necessary."""
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


class LowFrequencyMRF(nn.Module):
    """
    Low-frequency material-aware consistency fusion module.

    Low-frequency features usually contain region structure, object contour,
    background intensity and material base information. Therefore, this module
    uses a material-guided consistency factor to control the interaction between
    infrared and visible low-frequency features.

    Main idea:
        alpha_l = sigmoid(phi_l([low_ir, low_vis, |low_ir-low_vis|, z_m]))

        interactive = phi_i([low_ir, low_vis, low_ir * low_vis])
        base        = phi_b([low_ir, low_vis])

        low_fused = alpha_l * interactive + (1 - alpha_l) * base

    Ablation:
        - use_material=False:
            z_m is not used.
        - use_consistency=False:
            alpha_l is fixed to 1, so the module only uses interactive fusion.
    """

    def __init__(
        self,
        feature_channels: int = 64,
        material_channels: int = 64,
        hidden_channels: int = 64,
        use_material: bool = True,
        use_consistency: bool = True,
        return_aux: bool = True,
    ) -> None:
        super().__init__()

        self.feature_channels = feature_channels
        self.material_channels = material_channels
        self.hidden_channels = hidden_channels
        self.use_material = use_material
        self.use_consistency = use_consistency
        self.return_aux = return_aux

        # Project material prior to the same channel dimension as low features.
        if use_material:
            self.material_proj = nn.Sequential(
                nn.Conv2d(material_channels, feature_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature_channels),
                nn.PReLU(feature_channels),
            )
            consistency_in_channels = feature_channels * 4
        else:
            self.material_proj = None
            consistency_in_channels = feature_channels * 3

        # Estimate low-frequency cross-modal consistency factor alpha_l.
        self.consistency_estimator = nn.Sequential(
            ConvBNPReLU(consistency_in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Interactive branch: emphasizes coordinated cross-modal structure.
        self.interaction_branch = nn.Sequential(
            ConvBNPReLU(feature_channels * 3, hidden_channels, kernel_size=3, padding=1),
            ConvBNPReLU(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.PReLU(feature_channels),
        )

        # Base branch: preserves stable low-frequency information.
        self.base_branch = nn.Sequential(
            ConvBNPReLU(feature_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.PReLU(feature_channels),
        )

        # Final refinement after consistency-controlled fusion.
        self.refine = nn.Sequential(
            ConvBNPReLU(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        self.out_act = nn.PReLU(feature_channels)

    def forward(
        self,
        low_ir: torch.Tensor,
        low_vis: torch.Tensor,
        z_m: Optional[torch.Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Args:
            low_ir:
                [B, C, H, W], infrared low-frequency decoded feature.
            low_vis:
                [B, C, H, W], visible low-frequency decoded feature.
            z_m:
                [B, C_m, H_z, W_z], material prior.
            return_aux:
                Whether to return intermediate tensors.

        Returns:
            low_fused:
                [B, C, H, W]
            aux:
                Dictionary containing consistency map and intermediate features.
        """
        if low_ir.shape != low_vis.shape:
            raise ValueError(
                f"low_ir and low_vis must have the same shape, got "
                f"{low_ir.shape} and {low_vis.shape}."
            )
        if low_ir.dim() != 4:
            raise ValueError("low_ir and low_vis must be 4D tensors: [B, C, H, W].")
        if low_ir.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected feature_channels={self.feature_channels}, "
                f"but got low feature channels={low_ir.shape[1]}."
            )
        if self.use_material and z_m is None:
            raise ValueError("z_m must be provided when use_material=True.")

        if return_aux is None:
            return_aux = self.return_aux

        diff = torch.abs(low_ir - low_vis)
        prod = low_ir * low_vis

        # Low-frequency interactive feature.
        interactive = self.interaction_branch(torch.cat([low_ir, low_vis, prod], dim=1))

        # Stable base feature.

        # base = self.base_branch(torch.cat([low_ir, low_vis], dim=1))
        # 0521实验
        base = self.base_branch(torch.cat([low_vis, low_vis], dim=1))

        # Material-guided consistency estimation.
        material_feat: Optional[torch.Tensor] = None
        if self.use_material:
            material_feat = resize_like(z_m, low_ir)
            material_feat = self.material_proj(material_feat)
            consistency_input = torch.cat([low_ir, low_vis, diff, material_feat], dim=1)
        else:
            consistency_input = torch.cat([low_ir, low_vis, diff], dim=1)

        if self.use_consistency:
            consistency = self.consistency_estimator(consistency_input)
        else:
            consistency = torch.ones_like(low_ir)

        low_fused_raw = consistency * interactive + (1.0 - consistency) * base
        low_fused = self.out_act(self.refine(low_fused_raw) + low_fused_raw)

        aux: Dict[str, Optional[torch.Tensor]] = {}
        if return_aux:
            aux = {
                "low_consistency": consistency,
                "low_interactive": interactive,
                "low_base": base,
                "low_material_feat": material_feat,
                "low_fused_raw": low_fused_raw,
            }

        return low_fused, aux


class HighFrequencyMRF(nn.Module):
    """
    High-frequency material-aware reliability fusion module.

    High-frequency features usually contain edges, textures, thermal boundaries
    and small salient structures. Infrared and visible high-frequency components
    are relatively independent, so this module uses material-guided reliability
    weights to select more trustworthy modal details.

    Main idea:
        [w_ir, w_vis] = softmax(phi_h([high_ir, high_vis, |high_ir-high_vis|, z_m]))

        high_fused = w_ir * high_ir + w_vis * high_vis

    Ablation:
        - use_material=False:
            reliability is estimated without z_m.
        - use_reliability=False:
            fixed equal weights are used.
    """

    def __init__(
        self,
        feature_channels: int = 64,
        material_channels: int = 64,
        hidden_channels: int = 64,
        use_material: bool = True,
        use_reliability: bool = True,
        return_aux: bool = True,
    ) -> None:
        super().__init__()

        self.feature_channels = feature_channels
        self.high_channels = feature_channels * 3
        self.material_channels = material_channels
        self.hidden_channels = hidden_channels
        self.use_material = use_material
        self.use_reliability = use_reliability
        self.return_aux = return_aux

        # Project material prior to high-frequency channel dimension.
        if use_material:
            self.material_proj = nn.Sequential(
                nn.Conv2d(material_channels, self.high_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.high_channels),
                nn.PReLU(self.high_channels),
            )
            reliability_in_channels = self.high_channels * 4
        else:
            self.material_proj = None
            reliability_in_channels = self.high_channels * 3

        # Estimate two modal reliability logits.
        # Output channel = 2 * high_channels, then reshape to
        # [B, 2, high_channels, H, W] and apply softmax over modal dimension.
        self.reliability_estimator = nn.Sequential(
            ConvBNPReLU(reliability_in_channels, hidden_channels, kernel_size=3, padding=1),
            ConvBNPReLU(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, self.high_channels * 2, kernel_size=1, bias=True),
        )

        # Optional refinement after reliability-weighted high-frequency selection.
        self.refine = nn.Sequential(
            ConvBNPReLU(self.high_channels, self.high_channels, kernel_size=3, padding=1),
            nn.Conv2d(self.high_channels, self.high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.high_channels),
        )
        self.out_act = nn.PReLU(self.high_channels)

    def forward(
        self,
        high_ir: torch.Tensor,
        high_vis: torch.Tensor,
        z_m: Optional[torch.Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Args:
            high_ir:
                [B, 3C, H, W], infrared high-frequency decoded feature.
            high_vis:
                [B, 3C, H, W], visible high-frequency decoded feature.
            z_m:
                [B, C_m, H_z, W_z], material prior.
            return_aux:
                Whether to return intermediate tensors.

        Returns:
            high_fused:
                [B, 3C, H, W]
            aux:
                Dictionary containing reliability weights and intermediate features.
        """
        if high_ir.shape != high_vis.shape:
            raise ValueError(
                f"high_ir and high_vis must have the same shape, got "
                f"{high_ir.shape} and {high_vis.shape}."
            )
        if high_ir.dim() != 4:
            raise ValueError("high_ir and high_vis must be 4D tensors: [B, C, H, W].")
        if high_ir.shape[1] != self.high_channels:
            raise ValueError(
                f"Expected high feature channels={self.high_channels}, "
                f"but got high feature channels={high_ir.shape[1]}."
            )
        if self.use_material and z_m is None:
            raise ValueError("z_m must be provided when use_material=True.")

        if return_aux is None:
            return_aux = self.return_aux

        diff = torch.abs(high_ir - high_vis)

        material_feat: Optional[torch.Tensor] = None
        if self.use_material:
            material_feat = resize_like(z_m, high_ir)
            material_feat = self.material_proj(material_feat)
            reliability_input = torch.cat([high_ir, high_vis, diff, material_feat], dim=1)
        else:
            reliability_input = torch.cat([high_ir, high_vis, diff], dim=1)

        if self.use_reliability:
            B, _, H, W = high_ir.shape
            logits = self.reliability_estimator(reliability_input)
            logits = logits.view(B, 2, self.high_channels, H, W)
            weights = torch.softmax(logits, dim=1)

            w_ir = weights[:, 0, :, :, :]
            w_vis = weights[:, 1, :, :, :]
        else:
            w_ir = torch.full_like(high_ir, 0.5)
            w_vis = torch.full_like(high_vis, 0.5)

        high_fused_raw = w_ir * high_ir + w_vis * high_vis
        high_fused = self.out_act(self.refine(high_fused_raw) + high_fused_raw)

        aux: Dict[str, Optional[torch.Tensor]] = {}
        if return_aux:
            aux = {
                "high_reliability_ir": w_ir,
                "high_reliability_vis": w_vis,
                "high_material_feat": material_feat,
                "high_fused_raw": high_fused_raw,
            }

        return high_fused, aux


class MaterialAwareMRF(nn.Module):
    """
    Wrapper module that contains both LowFrequencyMRF and HighFrequencyMRF.

    This wrapper is optional. It is useful in the main fusion network because
    low-frequency and high-frequency fusion can be called together.
    """

    def __init__(
        self,
        feature_channels: int = 64,
        material_channels: int = 64,
        hidden_channels: int = 64,
        use_material: bool = True,
        use_low_mrf: bool = True,
        use_high_mrf: bool = True,
        use_low_consistency: bool = True,
        use_high_reliability: bool = True,
        return_aux: bool = True,
    ) -> None:
        super().__init__()

        self.feature_channels = feature_channels
        self.use_low_mrf = use_low_mrf
        self.use_high_mrf = use_high_mrf
        self.return_aux = return_aux

        self.low_mrf = LowFrequencyMRF(
            feature_channels=feature_channels,
            material_channels=material_channels,
            hidden_channels=hidden_channels,
            use_material=use_material,
            use_consistency=use_low_consistency,
            return_aux=return_aux,
        )

        self.high_mrf = HighFrequencyMRF(
            feature_channels=feature_channels,
            material_channels=material_channels,
            hidden_channels=hidden_channels,
            use_material=use_material,
            use_reliability=use_high_reliability,
            return_aux=return_aux,
        )

    def forward(
        self,
        low_ir: torch.Tensor,
        low_vis: torch.Tensor,
        high_ir: torch.Tensor,
        high_vis: torch.Tensor,
        z_m: Optional[torch.Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Args:
            low_ir, low_vis:
                [B, C, H_l, W_l]
            high_ir, high_vis:
                [B, 3C, H_h, W_h]
            z_m:
                [B, C_m, H_z, W_z]

        Returns:
            low_fused:
                [B, C, H_l, W_l]
            high_fused:
                [B, 3C, H_h, W_h]
            aux:
                {
                    "low": {...},
                    "high": {...}
                }
        """
        if return_aux is None:
            return_aux = self.return_aux

        if self.use_low_mrf:
            low_fused, low_aux = self.low_mrf(
                low_ir=low_ir,
                low_vis=low_vis,
                z_m=z_m,
                return_aux=return_aux,
            )
        else:
            low_fused = 0.5 * (low_ir + low_vis)
            low_aux = {}

        if self.use_high_mrf:
            high_fused, high_aux = self.high_mrf(
                high_ir=high_ir,
                high_vis=high_vis,
                z_m=z_m,
                return_aux=return_aux,
            )
        else:
            high_fused = 0.5 * (high_ir + high_vis)
            high_aux = {}

        aux = {
            "low": low_aux,
            "high": high_aux,
        }

        return low_fused, high_fused, aux


if __name__ == "__main__":
    import gc
    import time

    def format_mb(num_bytes):
        """Convert bytes to MB."""
        return num_bytes / 1024 / 1024

    def print_cuda_memory(title, device):
        """Print current and peak CUDA memory usage."""
        if device.type != "cuda":
            return

        torch.cuda.synchronize(device)

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)

        print(f"\n[{title}]")
        print(f"Current allocated memory : {format_mb(allocated):.2f} MB")
        print(f"Current reserved memory  : {format_mb(reserved):.2f} MB")
        print(f"Peak allocated memory    : {format_mb(max_allocated):.2f} MB")
        print(f"Peak reserved memory     : {format_mb(max_reserved):.2f} MB")

    def count_parameters(model):
        """Return total and trainable parameter numbers."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def warmup_cuda(
        model,
        low_ir,
        low_vis,
        high_ir,
        high_vis,
        z_m,
        iters=5,
    ):
        """Warm up CUDA kernels for more stable timing."""
        model.eval()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(
                    low_ir=low_ir,
                    low_vis=low_vis,
                    high_ir=high_ir,
                    high_vis=high_vis,
                    z_m=z_m,
                    return_aux=False,
                )

    # -----------------------------
    # 1. Device setting
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # -----------------------------
    # 2. Test input size
    # -----------------------------
    B, C = 2, 64

    # Decoder outputs before MRF.
    # Low-frequency decoded feature:  [B, C, H_l, W_l]
    # High-frequency decoded feature: [B, 3C, H_h, W_h]
    H_low, W_low = 64, 64
    H_high, W_high = 64, 64

    # Material prior z_m is usually generated at shallow feature scale.
    H_z, W_z = 128, 128

    low_ir = torch.randn(B, C, H_low, W_low, device=device)
    low_vis = torch.randn(B, C, H_low, W_low, device=device)

    high_ir = torch.randn(B, 3 * C, H_high, W_high, device=device)
    high_vis = torch.randn(B, 3 * C, H_high, W_high, device=device)

    z_m = torch.randn(B, C, H_z, W_z, device=device)

    # -----------------------------
    # 3. Build MRF module
    # -----------------------------
    mrf = MaterialAwareMRF(
        feature_channels=C,
        material_channels=C,
        hidden_channels=64,
        use_material=True,
        use_low_mrf=True,
        use_high_mrf=True,
        use_low_consistency=True,
        use_high_reliability=True,
        return_aux=True,
    ).to(device)

    total_params, trainable_params = count_parameters(mrf)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print_cuda_memory("After model and input initialization", device)

    # -----------------------------
    # 4. Inference memory and time test
    # -----------------------------
    mrf.eval()

    if device.type == "cuda":
        warmup_cuda(mrf, low_ir, low_vis, high_ir, high_vis, z_m, iters=5)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    infer_iters = 20

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.time()

        for _ in range(infer_iters):
            low_fused, high_fused, aux = mrf(
                low_ir=low_ir,
                low_vis=low_vis,
                high_ir=high_ir,
                high_vis=high_vis,
                z_m=z_m,
                return_aux=True,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end_time = time.time()

    print("\n========== Inference Test ==========")
    print("low_fused shape :", tuple(low_fused.shape))
    print("high_fused shape:", tuple(high_fused.shape))
    print(f"Average inference time: {(end_time - start_time) * 1000 / infer_iters:.2f} ms")

    print("\nLow aux:")
    for k, v in aux["low"].items():
        print(k, None if v is None else tuple(v.shape))

    print("\nHigh aux:")
    for k, v in aux["high"].items():
        print(k, None if v is None else tuple(v.shape))

    print_cuda_memory("Inference memory usage", device)

    # Clear inference tensors.
    del low_fused, high_fused, aux
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------
    # 5. Training forward + backward memory and time test
    # -----------------------------
    mrf.train()

    low_ir_train = torch.randn(
        B, C, H_low, W_low,
        device=device,
        requires_grad=True,
    )
    low_vis_train = torch.randn(
        B, C, H_low, W_low,
        device=device,
        requires_grad=True,
    )

    high_ir_train = torch.randn(
        B, 3 * C, H_high, W_high,
        device=device,
        requires_grad=True,
    )
    high_vis_train = torch.randn(
        B, 3 * C, H_high, W_high,
        device=device,
        requires_grad=True,
    )

    z_m_train = torch.randn(
        B, C, H_z, W_z,
        device=device,
        requires_grad=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()

    low_fused, high_fused, aux = mrf(
        low_ir=low_ir_train,
        low_vis=low_vis_train,
        high_ir=high_ir_train,
        high_vis=high_vis_train,
        z_m=z_m_train,
        return_aux=True,
    )

    # Dummy loss for backward memory test.
    loss = low_fused.mean() + high_fused.mean()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print_cuda_memory("Training forward memory usage", device)

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.time()

    print("\n========== Training Test ==========")
    print("low_fused shape :", tuple(low_fused.shape))
    print("high_fused shape:", tuple(high_fused.shape))
    print("loss:", float(loss.detach().cpu()))
    print(f"Forward + backward time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Training forward + backward memory usage", device)

    # -----------------------------
    # 6. Ablation shape sanity check
    # -----------------------------
    print("\n========== Ablation Shape Sanity Check ==========")

    ablation_settings = [
        {
            "name": "w/o material",
            "kwargs": dict(use_material=False),
            "z_input": None,
        },
        {
            "name": "w/o low MRF",
            "kwargs": dict(use_low_mrf=False),
            "z_input": z_m,
        },
        {
            "name": "w/o high MRF",
            "kwargs": dict(use_high_mrf=False),
            "z_input": z_m,
        },
        {
            "name": "w/o low consistency",
            "kwargs": dict(use_low_consistency=False),
            "z_input": z_m,
        },
        {
            "name": "w/o high reliability",
            "kwargs": dict(use_high_reliability=False),
            "z_input": z_m,
        },
    ]

    for item in ablation_settings:
        kwargs = dict(
            feature_channels=C,
            material_channels=C,
            hidden_channels=64,
            use_material=True,
            use_low_mrf=True,
            use_high_mrf=True,
            use_low_consistency=True,
            use_high_reliability=True,
            return_aux=False,
        )
        kwargs.update(item["kwargs"])

        model_ablation = MaterialAwareMRF(**kwargs).to(device)
        model_ablation.eval()

        with torch.no_grad():
            lf, hf, _ = model_ablation(
                low_ir=low_ir,
                low_vis=low_vis,
                high_ir=high_ir,
                high_vis=high_vis,
                z_m=item["z_input"],
                return_aux=False,
            )

        print(f"{item['name']:<22} low: {tuple(lf.shape)}, high: {tuple(hf.shape)}")

        del model_ablation, lf, hf
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------
    # 7. Final cleanup
    # -----------------------------
    mrf.zero_grad(set_to_none=True)

    del low_fused, high_fused, aux, loss
    del low_ir, low_vis, high_ir, high_vis, z_m
    del low_ir_train, low_vis_train
    del high_ir_train, high_vis_train
    del z_m_train
    del mrf

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print_cuda_memory("After cleanup", device)