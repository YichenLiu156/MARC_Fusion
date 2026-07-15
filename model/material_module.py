"""
Material information extraction module for MARCFusion.

This file implements a Downsampled Axial SSM + Depthwise Conv Local Branch
material prior extractor. It does not depend on third-party SSM/Mamba packages.

Expected input:
    e_ir:  [B, C, H, W]  shallow infrared feature E_IR^1
    e_vis: [B, C, H, W]  shallow visible feature E_VIS^1

Output:
    z_m: [B, out_channels, H, W] unified material prior
    aux: optional dictionary for visualization / ablation analysis
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    """Small Conv-BN-PReLU block used only inside the material module."""

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


class SelectiveAxisScan(nn.Module):
    """
    A simple pure-PyTorch 1D selective state scan along one spatial axis.

    For each sequence position t:
        h_t = a_t * h_{t-1} + b_t * x_t
        y_t = c_t * h_t + d * x_t

    Notes:
        - This is an SSM-like recurrent scan for visual feature maps.
        - It is intentionally lightweight and dependency-free.
        - The scan is performed with a Python loop, so it is best used after
          spatial downsampling.
    """

    def __init__(self, channels: int, axis: str = "width") -> None:
        super().__init__()
        if axis not in {"width", "height"}:
            raise ValueError("axis must be either 'width' or 'height'.")
        self.channels = channels
        self.axis = axis

        # Generate input-dependent state parameters A, B, C.
        self.param_proj = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)

        # Direct input path D. Initialized to 1 so the block starts close to an
        # identity-enhanced recurrent transformation.
        self.d = nn.Parameter(torch.ones(1, channels, 1, 1))

    def _get_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c = self.param_proj(x).chunk(3, dim=1)

        # Keep A in (0, 1) for stable memory retention.
        # A close to 1 means more historical information is preserved.
        a = 0.99 * torch.sigmoid(a)
        b = torch.sigmoid(b)
        c = torch.sigmoid(c)
        return a, b, c

    def _scan_width(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        h_state = x.new_zeros(B, C, H)
        outputs = []
        indices = range(W - 1, -1, -1) if reverse else range(W)

        d = self.d.squeeze(-1)  # [1, C, 1]
        for t in indices:
            x_t = x[:, :, :, t]
            a_t = a[:, :, :, t]
            b_t = b[:, :, :, t]
            c_t = c[:, :, :, t]
            h_state = a_t * h_state + b_t * x_t
            y_t = c_t * h_state + d * x_t
            outputs.append(y_t.unsqueeze(-1))

        if reverse:
            outputs = outputs[::-1]
        return torch.cat(outputs, dim=-1)

    def _scan_height(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        h_state = x.new_zeros(B, C, W)
        outputs = []
        indices = range(H - 1, -1, -1) if reverse else range(H)

        d = self.d  # [1, C, 1, 1]
        for t in indices:
            x_t = x[:, :, t, :]
            a_t = a[:, :, t, :]
            b_t = b[:, :, t, :]
            c_t = c[:, :, t, :]
            h_state = a_t * h_state + b_t * x_t
            y_t = c_t * h_state + d.squeeze(-1) * x_t
            outputs.append(y_t.unsqueeze(2))

        if reverse:
            outputs = outputs[::-1]
        return torch.cat(outputs, dim=2)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        a, b, c = self._get_params(x)
        if self.axis == "width":
            return self._scan_width(x, a, b, c, reverse=reverse)
        return self._scan_height(x, a, b, c, reverse=reverse)


class DownsampledAxialSSM(nn.Module):
    """
    Downsampled axial SSM block.

    It first reduces the spatial size, then performs horizontal and vertical
    state scans, and finally upsamples the long-range material feature back to
    the original feature size.
    """

    def __init__(
        self,
        channels: int,
        downsample_ratio: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if downsample_ratio < 1:
            raise ValueError("downsample_ratio must be >= 1.")

        self.channels = channels
        self.downsample_ratio = downsample_ratio
        self.bidirectional = bidirectional

        self.pre = ConvBNPReLU(channels, channels, kernel_size=3, padding=1)

        self.scan_h = SelectiveAxisScan(channels, axis="width")
        self.scan_v = SelectiveAxisScan(channels, axis="height")

        # Number of directional outputs to fuse.
        num_paths = 4 if bidirectional else 2
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_paths, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.PReLU(channels)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_ratio == 1:
            return x
        _, _, H, W = x.shape
        Hs = max(1, math.ceil(H / self.downsample_ratio))
        Ws = max(1, math.ceil(W / self.downsample_ratio))
        return F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_down = self._downsample(x)
        x_down = self.pre(x_down)

        outputs = [
            self.scan_h(x_down, reverse=False),
            self.scan_v(x_down, reverse=False),
        ]
        if self.bidirectional:
            outputs.extend([
                self.scan_h(x_down, reverse=True),
                self.scan_v(x_down, reverse=True),
            ])

        y = self.fuse(torch.cat(outputs, dim=1))
        y = self.act(y + x_down)

        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class DepthwiseLocalBranch(nn.Module):
    """Local material compensation branch based on depthwise convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaterialPriorExtractor(nn.Module):
    """
    Material prior extractor M.

    This module is designed to be placed after shallow feature extraction and
    before DWT. It jointly models E_IR^1 and E_VIS^1 and outputs a unified
    material prior z_m for the later MRF modules.

    Modes:
        - "downsampled_ssm": global Downsampled Axial SSM + local DWConv branch.
        - "conv": local convolution branch only, useful for w/o SSM ablation.
        - "ssm_only": global Downsampled Axial SSM only, useful for w/o local branch.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 64,
        mode: str = "downsampled_ssm",
        downsample_ratio: int = 2,
        bidirectional: bool = True,
        return_aux: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"downsampled_ssm", "conv", "ssm_only"}:
            raise ValueError("mode must be 'downsampled_ssm', 'conv', or 'ssm_only'.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mode = mode
        self.return_aux = return_aux

        # Joint shallow material feature from infrared and visible features.
        self.joint_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
            ConvBNPReLU(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )

        self.global_branch = DownsampledAxialSSM(
            channels=hidden_channels,
            downsample_ratio=downsample_ratio,
            bidirectional=bidirectional,
        )
        self.local_branch = DepthwiseLocalBranch(hidden_channels)

        # Gate for fusing global material continuity and local material details.
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # self.material_head = nn.Sequential(
        #     ConvBNPReLU(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        #     nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        # )
        if out_channels % 2 != 0:
            raise ValueError(
                "out_channels must be even because z_m is split into z_r and z_e."
            )

        self.zr_channels = out_channels // 2
        self.ze_channels = out_channels // 2

        # VIS 偏向分支：用于给 z_r 提供可见光反射相关信息
        self.vis_bias_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
        )

        # IR 偏向分支：用于给 z_e 提供红外热发射相关信息
        self.ir_bias_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
        )

        # z_r：反射相关材质子空间，偏向 VIS
        self.reflective_head = nn.Sequential(
            ConvBNPReLU(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, self.zr_channels, kernel_size=1, bias=True),
        )

        # z_e：热发射相关材质子空间，偏向 IR
        self.emissive_head = nn.Sequential(
            ConvBNPReLU(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, self.ze_channels, kernel_size=1, bias=True),
        )

    def forward(
        self,
        e_ir: torch.Tensor,
        e_vis: torch.Tensor,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Args:
            e_ir:  [B, C, H, W], shallow infrared feature E_IR^1.
            e_vis: [B, C, H, W], shallow visible feature E_VIS^1.
            return_aux: whether to return intermediate features for debugging.

        Returns:
            z_m: [B, out_channels, H, W], unified material prior.
            aux: intermediate tensors. Empty if return_aux=False.
        """
        if e_ir.shape != e_vis.shape:
            raise ValueError(
                f"e_ir and e_vis must have the same shape, got {e_ir.shape} and {e_vis.shape}."
            )
        if e_ir.dim() != 4:
            raise ValueError("e_ir and e_vis must be 4D tensors: [B, C, H, W].")

        if return_aux is None:
            return_aux = self.return_aux

        joint = self.joint_proj(torch.cat([e_ir, e_vis], dim=1))

        global_feat: Optional[torch.Tensor]
        local_feat: Optional[torch.Tensor]
        gate: Optional[torch.Tensor]

        if self.mode == "conv":
            global_feat = None
            local_feat = self.local_branch(joint)
            fused = local_feat
            gate = None
        elif self.mode == "ssm_only":
            global_feat = self.global_branch(joint)
            local_feat = None
            fused = global_feat
            gate = None
        else:
            global_feat = self.global_branch(joint)
            local_feat = self.local_branch(joint)
            gate = self.gate(torch.cat([global_feat, local_feat], dim=1))
            fused = gate * global_feat + (1.0 - gate) * local_feat

        # z_m = self.material_head(fused)
        vis_bias = self.vis_bias_proj(e_vis)
        ir_bias = self.ir_bias_proj(e_ir)

        z_r = self.reflective_head(
            torch.cat([fused, vis_bias], dim=1)
        )

        z_e = self.emissive_head(
            torch.cat([fused, ir_bias], dim=1)
        )

        z_m = torch.cat([z_r, z_e], dim=1)

        aux: Dict[str, Optional[torch.Tensor]] = {}
        if return_aux:
            aux = {
                "joint_material_feat": joint,
                "global_material_feat": global_feat,
                "local_material_feat": local_feat,
                "material_gate": gate,
                "fused_material_feat": fused,
            }
            if z_m.shape[1] % 2 == 0:
                # z_reflective, z_emissive = torch.chunk(z_m, chunks=2, dim=1)
                # aux["z_reflective"] = z_reflective
                # aux["z_emissive"] = z_emissive
                aux["z_reflective"] = z_r
                aux["z_emissive"] = z_e
            else:
                aux["z_reflective"] = None
                aux["z_emissive"] = None

        return z_m, aux


# if __name__ == "__main__":
#     # Quick shape test.
#     B, C, H, W = 2, 64, 128, 128
#     e_ir = torch.randn(B, C, H, W)
#     e_vis = torch.randn(B, C, H, W)
#
#     module = MaterialPriorExtractor(
#         in_channels=C,
#         hidden_channels=64,
#         out_channels=64,
#         mode="downsampled_ssm",
#         downsample_ratio=2,
#         bidirectional=True,
#         return_aux=True,
#     )
#     z_m, aux = module(e_ir, e_vis)
#     print("z_m:", z_m.shape)
#     for k, v in aux.items():
#         print(k, None if v is None else tuple(v.shape))
if __name__ == "__main__":
    import gc
    import time

    def format_mb(x: int) -> float:
        return x / 1024 / 1024

    def print_cuda_memory(title: str, device: torch.device) -> None:
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
    B, C, H, W = 2, 64, 128, 128

    e_ir = torch.randn(B, C, H, W, device=device)
    e_vis = torch.randn(B, C, H, W, device=device)

    # -----------------------------
    # 3. Build module
    # -----------------------------
    module = MaterialPriorExtractor(
        in_channels=C,
        hidden_channels=64,
        out_channels=64,
        mode="downsampled_ssm",
        downsample_ratio=2,
        bidirectional=True,
        return_aux=True,
    ).to(device)

    print_cuda_memory("After model and input initialization", device)

    # -----------------------------
    # 4. Inference memory test
    # -----------------------------
    module.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        start_time = time.time()
        z_m, aux = module(e_ir, e_vis)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_time = time.time()

    print("\n========== Inference Test ==========")
    print("z_m shape:", tuple(z_m.shape))
    for k, v in aux.items():
        print(k, None if v is None else tuple(v.shape))
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Inference memory usage", device)

    # 清理推理阶段变量
    del z_m, aux
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------
    # 5. Training memory test
    # -----------------------------
    module.train()

    e_ir_train = torch.randn(B, C, H, W, device=device, requires_grad=True)
    e_vis_train = torch.randn(B, C, H, W, device=device, requires_grad=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    z_m, aux = module(e_ir_train, e_vis_train)

    # 构造一个简单 loss，用于测试 backward 显存
    loss = z_m.mean()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print_cuda_memory("Training forward memory usage", device)

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.time()

    print("\n========== Training Test ==========")
    print("z_m shape:", tuple(z_m.shape))
    print("loss:", float(loss.detach().cpu()))
    print(f"Forward + backward time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Training forward + backward memory usage", device)

    # -----------------------------
    # 6. Final cleanup
    # -----------------------------
    module.zero_grad(set_to_none=True)

    del z_m, aux, loss
    del e_ir, e_vis, e_ir_train, e_vis_train, module

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print_cuda_memory("After cleanup", device)