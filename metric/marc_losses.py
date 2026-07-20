"""Losses used by the two-stage MARC-Fusion training scheme.

Material pretraining:
    L_mat = lambda_rec * L_rec
          + lambda_mg  * L_mgrad
          + lambda_c   * L_msc

Fusion training:
    L_fus = lambda_int * L_int
          + lambda_fg  * L_fgrad

"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_arg(args: Any, name: str, default: Any) -> Any:
    """Read an option from an argparse namespace or a dictionary."""
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def get_first_arg(args: Any, names: Sequence[str], default: Any) -> Any:
    """Read the first available option from a sequence of equivalent names."""
    sentinel = object()
    for name in names:
        value = get_arg(args, name, sentinel)
        if value is not sentinel:
            return value
    return default


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    """Compute a channel-wise Sobel gradient magnitude map."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

    channels = x.shape[1]
    sobel_x = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = x.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)

    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(x, sobel_x, groups=channels)
    grad_y = F.conv2d(x, sobel_y, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def _require_same_shape(*tensors: torch.Tensor) -> None:
    shapes = [tuple(t.shape) for t in tensors]
    if len(set(shapes)) != 1:
        raise ValueError(f"All image tensors must have the same shape, got {shapes}")


def _material_representation(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:

    for key in ("p_m", "proj_z_m", "projected_z_m", "z_m"):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(
        "Material outputs must contain one of: p_m, proj_z_m, "
        "projected_z_m, z_m."
    )


def _normalized_vector(z: torch.Tensor) -> torch.Tensor:
    """Convert a feature map/vector to a normalized per-sample vector."""
    if z.ndim == 4:
        z = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
    elif z.ndim == 3:
        z = z.mean(dim=-1)
    elif z.ndim != 2:
        raise ValueError(
            "Material representation must be [B, C], [B, C, N], or "
            f"[B, C, H, W], got {tuple(z.shape)}"
        )
    return F.normalize(z, dim=1)



class ImageReconstructionLoss(nn.Module):
    """L_rec: reconstruct the infrared and visible inputs."""

    def forward(
        self,
        recon_ir: torch.Tensor,
        recon_vis: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        _require_same_shape(recon_ir, ir)
        _require_same_shape(recon_vis, vis)
        return F.l1_loss(recon_ir, ir) + F.l1_loss(recon_vis, vis)


class GradientReconstructionLoss(nn.Module):
    """L_mgrad: preserve material-related boundaries and local structures."""

    def forward(
        self,
        recon_ir: torch.Tensor,
        recon_vis: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        _require_same_shape(recon_ir, ir)
        _require_same_shape(recon_vis, vis)
        return F.l1_loss(gradient_map(recon_ir), gradient_map(ir)) + F.l1_loss(
            gradient_map(recon_vis), gradient_map(vis)
        )


class StabilityContrastiveLoss(nn.Module):
    """L_msc from equation (18).

    L_msc = log(1 + exp((s_n - s_p) / tau))

    s_p is the cosine similarity between two independently perturbed views of
    the same registered pair; s_n is the cosine similarity to one negative
    patch pair sampled from a different registered image pair.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
    ) -> torch.Tensor:
        anchor = _normalized_vector(z_anchor)
        positive = _normalized_vector(z_positive)
        negative = _normalized_vector(z_negative)

        if positive.shape != anchor.shape:
            raise ValueError(
                "Anchor and positive representations must have the same shape, "
                f"got {tuple(anchor.shape)} and {tuple(positive.shape)}"
            )

        if negative.shape[0] == 1 and anchor.shape[0] > 1:
            negative = negative.expand(anchor.shape[0], -1)
        if negative.shape != anchor.shape:
            raise ValueError(
                "Each anchor must have one corresponding negative representation, "
                f"got {tuple(anchor.shape)} and {tuple(negative.shape)}"
            )

        positive_similarity = (anchor * positive).sum(dim=1)
        negative_similarity = (anchor * negative).sum(dim=1)

        return F.softplus(
            (negative_similarity - positive_similarity) / self.temperature
        ).mean()


class MaterialStageLoss(nn.Module):

    def __init__(
        self,
        recon_weight: float = 1.0,
        grad_weight: float = 0.5,
        contrast_weight: float = 0.1,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.grad_weight = grad_weight
        self.contrast_weight = contrast_weight

        self.reconstruction = ImageReconstructionLoss()
        self.gradient_reconstruction = GradientReconstructionLoss()
        self.stability_contrastive = StabilityContrastiveLoss(temperature)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        ir: torch.Tensor,
        vis: torch.Tensor,
        outputs_aug1: Optional[Dict[str, torch.Tensor]] = None,
        outputs_aug2: Optional[Dict[str, torch.Tensor]] = None,
        outputs_neg: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        recon_ir = outputs["recon_ir"]
        recon_vis = outputs["recon_vis"]

        loss_recon = self.reconstruction(recon_ir, recon_vis, ir, vis)
        loss_grad = self.gradient_reconstruction(recon_ir, recon_vis, ir, vis)

        if self.contrast_weight > 0:
            if outputs_aug1 is None or outputs_aug2 is None or outputs_neg is None:
                raise ValueError(
                    "L_msc requires outputs_aug1, outputs_aug2, and outputs_neg."
                )
            loss_msc = self.stability_contrastive(
                _material_representation(outputs_aug1),
                _material_representation(outputs_aug2),
                _material_representation(outputs_neg),
            )
        else:
            loss_msc = recon_ir.new_zeros(())

        total = (
            self.recon_weight * loss_recon
            + self.grad_weight * loss_grad
            + self.contrast_weight * loss_msc
        )

        return total, {
            "loss_total": total.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_grad_recon": loss_grad.detach(),
            "loss_msc": loss_msc.detach(),
        }


def build_material_stage_loss(args: Any = None) -> MaterialStageLoss:
    return MaterialStageLoss(
        recon_weight=get_arg(args, "recon_weight", 1.0),
        grad_weight=get_arg(args, "grad_weight", 0.5),
        contrast_weight=get_first_arg(
            args,
            ("material_contrast_weight", "material_consistency_weight"),
            0.1,
        ),
        temperature=get_first_arg(
            args,
            ("material_temperature", "negative_temperature"),
            0.2,
        ),
    )




class FusionIntensityLoss(nn.Module):
    """L_int: combine salient source intensity and average modality intensity."""

    def __init__(self, salient_weight: float = 0.6) -> None:
        super().__init__()
        if not 0.0 <= salient_weight <= 1.0:
            raise ValueError("salient_weight must be in [0, 1]")
        self.salient_weight = salient_weight

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        _require_same_shape(fused, ir, vis)
        salient = torch.maximum(ir, vis)
        average = 0.5 * (ir + vis)
        target = self.salient_weight * salient + (
            1.0 - self.salient_weight
        ) * average
        return F.l1_loss(fused, target)


class FusionGradientLoss(nn.Module):
    """L_fgrad: preserve the strongest source-image structural gradients."""

    def forward(
        self,
        fused: torch.Tensor,
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        _require_same_shape(fused, ir, vis)
        target_gradient = torch.maximum(gradient_map(ir), gradient_map(vis))
        return F.l1_loss(gradient_map(fused), target_gradient)


class FusionStageLoss(nn.Module):

    def __init__(
        self,
        intensity_weight: float = 0.35,
        gradient_weight: float = 1.20,
        salient_weight: float = 0.58,
    ) -> None:
        super().__init__()
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.intensity = FusionIntensityLoss(salient_weight=salient_weight)
        self.gradient = FusionGradientLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fused = outputs["fused"]
        loss_intensity = self.intensity(fused, ir, vis)
        loss_gradient = self.gradient(fused, ir, vis)

        total = (
            self.intensity_weight * loss_intensity
            + self.gradient_weight * loss_gradient
        )

        return total, {
            "loss_total": total.detach(),
            "loss_intensity": loss_intensity.detach(),
            "loss_gradient": loss_gradient.detach(),
        }


def build_fusion_stage_loss(args: Any = None) -> FusionStageLoss:
    return FusionStageLoss(
        intensity_weight=get_arg(args, "fusion_intensity_weight", 0.35),
        gradient_weight=get_arg(args, "fusion_gradient_weight", 1.20),
        salient_weight=get_first_arg(
            args,
            ("fusion_salient_weight", "fusion_intensity_alpha"),
            0.58,
        ),
    )


__all__ = [
    "gradient_map",
    "ImageReconstructionLoss",
    "GradientReconstructionLoss",
    "StabilityContrastiveLoss",
    "MaterialStageLoss",
    "build_material_stage_loss",
    "FusionIntensityLoss",
    "FusionGradientLoss",
    "FusionStageLoss",
    "build_fusion_stage_loss",
]
