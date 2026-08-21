import os

import cv2
import torch
# from setuptools.sandbox import save_path
from torchvision.utils import save_image
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
from torch import nn
from pytorch_wavelets import DWTInverse

# channel =64

output_feature = "./feature_figs"
# output_feature = None
color_map = cv2.COLORMAP_JET



class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act='prelu'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act == 'prelu':
            layers.append(nn.PReLU())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported activation: {act}")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    """
    使用 stride=2 卷积做下采样，避免额外池化带来的信息损失。
    """
    def __init__(self, in_channels, out_channels, act='prelu'):
        super().__init__()
        self.block = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            k=3, s=2, p=1,
            act=act
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class ResidualBlock(nn.Module):
    def __init__(self, channels, act='prelu'):
        super().__init__()
        if act == 'prelu':
            activation = nn.PReLU
        elif act == 'relu':
            activation = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = activation()

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = activation()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act2(out)
        return out

class FDEncoder(nn.Module):
    """
    目标：
    1. Stem 提取浅层特征
    2. 对浅层特征做 DWT
    3. 低频分支：2 次下采样
    4. 高频分支：保留三方向结构，仅做 1 次下采样
    5. 返回后续 decoder / fusion / IDWT 所需的多尺度特征

    返回说明：
    - stem_feat: 浅层特征，尺寸 H x W
    - low_feats: 低频多层特征字典
    - high_feats: 高频多层特征字典
    - yl: 原始 DWT 低频，尺寸 H/2 x W/2
    - yh: 原始 DWT 高频，尺寸 [B, C, 3, H/2, W/2]
    """
    def __init__(self, in_channels=1, base_channels=64, wave='haar'):
        super(FDEncoder, self).__init__()
        self.base_channels = base_channels
        self.dwt = DWTForward(wave=wave, J=1)

        # ---------- Stem ----------
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.PReLU(),
            ResidualBlock(base_channels, act='prelu'),
        )

        # ---------- Low-frequency branch ----------
        # 输入：yl, shape [B, C, H/2, W/2]
        self.low_stage0 = nn.Sequential(
            ConvBNAct(base_channels, base_channels, k=3, s=1, p=1, act='prelu'),
            ResidualBlock(base_channels, act='prelu'),
        )

        # 第1次下采样: H/2 -> H/4
        self.low_down1 = DownsampleBlock(base_channels, base_channels, act='prelu')
        self.low_stage1 = nn.Sequential(
            ResidualBlock(base_channels, act='prelu'),
            ResidualBlock(base_channels, act='prelu'),
        )

        # 第2次下采样: H/4 -> H/8
        self.low_down2 = DownsampleBlock(base_channels, base_channels, act='prelu')
        self.low_stage2 = nn.Sequential(
            ResidualBlock(base_channels, act='prelu'),
            ConvBNAct(base_channels, base_channels, k=3, s=1, p=1, act='tanh'),
        )

        # ---------- High-frequency branch ----------
        # 输入：yh, shape [B, C, 3, H/2, W/2]
        # 先 reshape 到 [B, 3C, H/2, W/2]，不压缩方向信息
        self.high_stage0 = nn.Sequential(
            ConvBNAct(base_channels * 3, base_channels * 3, k=3, s=1, p=1, act='prelu'),
            ResidualBlock(base_channels * 3, act='prelu'),
        )

        # 高频仅下采样 1 次: H/2 -> H/4
        self.high_down1 = DownsampleBlock(base_channels * 3, base_channels * 3, act='prelu')
        self.high_stage1 = nn.Sequential(
            ResidualBlock(base_channels * 3, act='prelu'),
            ConvBNAct(base_channels * 3, base_channels * 3, k=3, s=1, p=1, act='tanh'),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]

        Returns:
            stem_feat: [B, C, H, W]
            low_feats: {
                'l0': [B, C, H/2, W/2],
                'l1': [B, C, H/4, W/4],
                'l2': [B, C, H/8, W/8],
            }
            high_feats: {
                'h0': [B, 3C, H/2, W/2],
                'h1': [B, 3C, H/4, W/4],
            }
            yl: [B, C, H/2, W/2]
            yh: [B, C, 3, H/2, W/2]
        """
        # 1) 浅层特征
        stem_feat = self.stem(x)  # [B, C, H, W]

        # 2) DWT
        yl, yh_list = self.dwt(stem_feat)
        yh = yh_list[0]  # [B, C, 3, H/2, W/2]

        # ---------------- Low branch ----------------
        l0 = self.low_stage0(yl)           # [B, C, H/2, W/2]
        l1 = self.low_down1(l0)            # [B, C, H/4, W/4]
        l1 = self.low_stage1(l1)
        l2 = self.low_down2(l1)            # [B, C, H/8, W/8]
        l2 = self.low_stage2(l2)

        # ---------------- High branch ----------------
        # 保留三方向，不做 sum / abs / squeeze
        b, c, d, h, w = yh.shape
        assert d == 3, f"Expected 3 high-frequency subbands, got {d}"

        # [B, C, 3, H/2, W/2] -> [B, 3C, H/2, W/2]
        h0_in = yh.view(b, c * 3, h, w)
        h0 = self.high_stage0(h0_in)       # [B, 3C, H/2, W/2]
        h1 = self.high_down1(h0)           # [B, 3C, H/4, W/4]
        h1 = self.high_stage1(h1)

        low_feats = {
            'l0': l0,
            'l1': l1,
            'l2': l2,
        }

        high_feats = {
            'h0': h0,
            'h1': h1,
        }

        return stem_feat, low_feats, high_feats, yl, yh

class FDDecoder(nn.Module):
    """
    双模式 FDDecoder:

    mode='reconstruct':
        分别解码 IR / VIS，用于还原两幅原始图像

    mode='fusion':
        解码融合后的 low_feats / high_feats，用于后续 IDWT 生成融合图像
    """
    def __init__(self, channels=64):
        super(FDDecoder, self).__init__()
        self.channels = channels

        # Low-frequency decoder
        self.low_up1 = UpBlock(channels, channels, channels)
        self.low_up2 = UpBlock(channels, channels, channels)
        self.low_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )

        # High-frequency decoder
        self.high_up1 = UpBlock(channels * 3, channels * 3, channels * 3)
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.PReLU(),
        )

    def decode_single(self, low_feats, high_feats):
        """
        解码单一路径：IR、VIS 或融合特征都可以走这里。
        """
        l0, l1, l2 = low_feats["l0"], low_feats["l1"], low_feats["l2"]
        h0, h1 = high_feats["h0"], high_feats["h1"]

        low = self.low_up1(l2, l1)
        low = self.low_up2(low, l0)
        low_out = self.low_refine(low)

        high = self.high_up1(h1, h0)
        high_out = self.high_refine(high)

        return low_out, high_out

    def forward(
        self,
        low_feats_ir=None,
        high_feats_ir=None,
        low_feats_vis=None,
        high_feats_vis=None,
        low_feats_fused=None,
        high_feats_fused=None,
        mode="reconstruct",
    ):
        if mode == "reconstruct":
            assert low_feats_ir is not None and high_feats_ir is not None
            assert low_feats_vis is not None and high_feats_vis is not None

            low_ir, high_ir = self.decode_single(low_feats_ir, high_feats_ir)
            low_vis, high_vis = self.decode_single(low_feats_vis, high_feats_vis)

            return {
                "ir": {
                    "low": low_ir,
                    "high": high_ir,
                },
                "vis": {
                    "low": low_vis,
                    "high": high_vis,
                }
            }

        elif mode == "fusion":
            assert low_feats_fused is not None and high_feats_fused is not None

            low_fused, high_fused = self.decode_single(
                low_feats_fused,
                high_feats_fused
            )

            return {
                "fused": {
                    "low": low_fused,
                    "high": high_fused,
                }
            }

        else:
            raise ValueError("mode must be 'reconstruct' or 'fusion'")






class OutputResidualRefiner(nn.Module):
    """Lightweight three-band residual output reconstruction.

    The module consumes only MARC-fused representations and the base IDWT
    logit. It never reads the original IR/VIS images, so it cannot introduce a
    second cross-modal decision path.

    The last convolution predicts three reconstruction components:

        detail:   U_d - B_outer(U_d)
        mid:      B_inner(U_m) - B_outer(U_m)
        contrast: B_outer(U_c) - global_mean(B_outer(U_c))

    The original one-band SOTA17 checkpoint can be migrated exactly by copying
    its output kernel into the first channel and zero-initializing the other
    two channels. With detail_scale=0.25 and outer_kernel_size=15, the initial
    output then matches the SOTA17 output exactly.
    """

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        out_channels: int = 1,
        context_channels: int = 8,
        hidden_channels: int = 16,
        mid_inner_kernel_size: int = 3,
        outer_kernel_size: int = 15,
        detail_scale: float = 0.25,
        mid_scale: float = 0.10,
        contrast_scale: float = 0.05,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()

        for name, value in {
            "mid_inner_kernel_size": mid_inner_kernel_size,
            "outer_kernel_size": outer_kernel_size,
        }.items():
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 3.")
        if mid_inner_kernel_size >= outer_kernel_size:
            raise ValueError(
                "mid_inner_kernel_size must be smaller than "
                "outer_kernel_size."
            )
        for name, value in {
            "detail_scale": detail_scale,
            "mid_scale": mid_scale,
            "contrast_scale": contrast_scale,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if output_init_std <= 0.0:
            raise ValueError("output_init_std must be positive.")

        self.mid_inner_kernel_size = int(mid_inner_kernel_size)
        self.outer_kernel_size = int(outer_kernel_size)
        self.detail_scale = float(detail_scale)
        self.mid_scale = float(mid_scale)
        self.contrast_scale = float(contrast_scale)
        self.out_channels = int(out_channels)

        self.low_projection = nn.Conv2d(
            low_channels,
            context_channels,
            kernel_size=1,
            bias=True,
        )
        self.high_projection = nn.Conv2d(
            high_channels,
            context_channels,
            kernel_size=1,
            bias=True,
        )

        input_channels = out_channels + 2 * context_channels
        self.input_pad = nn.ReflectionPad2d(1)
        self.input_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            padding=0,
            bias=True,
        )
        self.activation = nn.PReLU(hidden_channels)
        self.output_pad = nn.ReflectionPad2d(1)
        self.output_conv = nn.Conv2d(
            hidden_channels,
            out_channels * 3,
            kernel_size=3,
            padding=0,
            bias=False,
        )

        nn.init.normal_(
            self.output_conv.weight,
            mean=0.0,
            std=output_init_std,
        )

    @staticmethod
    def _box_mean(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        if x.shape[-2] <= pad or x.shape[-1] <= pad:
            return F.avg_pool2d(
                x,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            )
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(
            x_pad,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        base_logit: torch.Tensor,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ):
        # The residual predictor must not back-propagate through MARC features.
        # The original base logit is nevertheless retained in refined_logit,
        # allowing the output low-frequency head to receive a controlled,
        # direct reconstruction gradient when it is explicitly unfrozen.
        base_logit_context = base_logit.detach()
        low_feature = low_feature.detach()
        high_feature = high_feature.detach()

        target_size = base_logit.shape[-2:]
        low_context = F.interpolate(
            self.low_projection(low_feature),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        high_context = F.interpolate(
            self.high_projection(high_feature),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        refiner_input = torch.cat(
            [base_logit_context, low_context, high_context],
            dim=1,
        )
        hidden = self.input_conv(self.input_pad(refiner_input))
        hidden = self.activation(hidden)
        residual_raw = self.output_conv(self.output_pad(hidden))
        detail_raw, mid_raw, contrast_raw = torch.chunk(
            residual_raw,
            chunks=3,
            dim=1,
        )

        detail_band = detail_raw - self._box_mean(
            detail_raw,
            self.outer_kernel_size,
        )
        mid_band = (
            self._box_mean(mid_raw, self.mid_inner_kernel_size)
            - self._box_mean(mid_raw, self.outer_kernel_size)
        )
        contrast_low = self._box_mean(
            contrast_raw,
            self.outer_kernel_size,
        )
        contrast_band = contrast_low - contrast_low.flatten(2).mean(
            dim=2,
            keepdim=True,
        ).unsqueeze(-1)

        detail_logit = self.detail_scale * torch.tanh(detail_band)
        mid_logit = self.mid_scale * torch.tanh(mid_band)
        contrast_logit = self.contrast_scale * torch.tanh(contrast_band)
        residual_logit = detail_logit + mid_logit + contrast_logit

        refined_logit = base_logit + residual_logit
        base_image = torch.sigmoid(base_logit)
        refined_image = torch.sigmoid(refined_logit)
        image_residual = refined_image - base_image

        eps = refined_image.new_tensor(1e-6)
        base_image_mean = (
            base_image.detach().abs().flatten(1).mean(dim=1) + eps
        )
        base_logit_mean = (
            base_logit.detach().abs().flatten(1).mean(dim=1) + eps
        )

        def image_ratio(value: torch.Tensor) -> torch.Tensor:
            return (
                value.abs().flatten(1).mean(dim=1) / base_image_mean
            ).mean()

        def logit_ratio(value: torch.Tensor) -> torch.Tensor:
            return (
                value.abs().flatten(1).mean(dim=1) / base_logit_mean
            ).mean()

        return {
            "image": refined_image,
            "base_image": base_image,
            "base_logit": base_logit,
            "refined_logit": refined_logit,
            "residual_raw": residual_raw,
            "residual_detail": detail_band,
            "residual_mid": mid_band,
            "residual_contrast": contrast_band,
            "residual_logit": residual_logit,
            "residual_detail_logit": detail_logit,
            "residual_mid_logit": mid_logit,
            "residual_contrast_logit": contrast_logit,
            "image_residual": image_residual,
            "image_residual_ratio": image_ratio(image_residual),
            "logit_residual_ratio": logit_ratio(residual_logit),
            "detail_logit_ratio": logit_ratio(detail_logit),
            "mid_logit_ratio": logit_ratio(mid_logit),
            "contrast_logit_ratio": logit_ratio(contrast_logit),
            "image_residual_abs_mean": image_residual.abs().mean(),
            "logit_residual_abs_mean": residual_logit.abs().mean(),
            "detail_logit_abs_mean": detail_logit.abs().mean(),
            "mid_logit_abs_mean": mid_logit.abs().mean(),
            "contrast_logit_abs_mean": contrast_logit.abs().mean(),
        }


class OutputLayers(nn.Module):
    """Convert decoded low/high features to reconstructed images.

    The original low/high heads and IDWT path are unchanged. The optional
    ``OutputResidualRefiner`` is applied only to the fused image after the base
    IDWT logit has been produced.
    """

    def __init__(
        self,
        channels=64,
        wave='haar',
        out_channels=1,
        output_high_scale=0.5,
        limit_output_high=True,
        use_output_bn=False,
        use_output_calibration=False,
        output_calibration_gain_range=0.35,
        output_calibration_bias_range=0.05,
        output_calibration_center=0.5,
        use_output_residual_refiner=False,
        output_residual_context_channels=8,
        output_residual_hidden_channels=16,
        output_residual_mid_inner_kernel=3,
        output_residual_outer_kernel=15,
        output_residual_detail_scale=0.25,
        output_residual_mid_scale=0.10,
        output_residual_contrast_scale=0.05,
        output_residual_init_std=1e-3,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels
        self.output_high_scale = float(output_high_scale)
        self.limit_output_high = bool(limit_output_high)
        self.use_output_bn = bool(use_output_bn)

        self.use_output_calibration = bool(use_output_calibration)
        self.output_calibration_gain_range = float(
            output_calibration_gain_range
        )
        self.output_calibration_bias_range = float(
            output_calibration_bias_range
        )
        self.output_calibration_center = float(output_calibration_center)

        if self.output_calibration_gain_range < 0.0:
            raise ValueError(
                "output_calibration_gain_range must be non-negative."
            )
        if self.output_calibration_bias_range < 0.0:
            raise ValueError(
                "output_calibration_bias_range must be non-negative."
            )

        if self.use_output_calibration:
            self.output_calibration_gain_raw = nn.Parameter(torch.zeros(1))
            self.output_calibration_bias_raw = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("output_calibration_gain_raw", None)
            self.register_parameter("output_calibration_bias_raw", None)

        self.idwt = DWTInverse(wave=wave)

        low_layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=0),
        ]
        if self.use_output_bn:
            low_layers.append(nn.BatchNorm2d(out_channels))
        self.low_head = nn.Sequential(*low_layers)

        high_layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels * 3,
                out_channels * 3,
                kernel_size=3,
                padding=0,
            ),
        ]
        if self.use_output_bn:
            high_layers.append(nn.BatchNorm2d(out_channels * 3))
        self.high_head = nn.Sequential(*high_layers)

        self.out_act = nn.Sigmoid()

        self.use_output_residual_refiner = bool(
            use_output_residual_refiner
        )
        if self.use_output_residual_refiner:
            self.output_residual_refiner = OutputResidualRefiner(
                low_channels=channels,
                high_channels=channels * 3,
                out_channels=out_channels,
                context_channels=output_residual_context_channels,
                hidden_channels=output_residual_hidden_channels,
                mid_inner_kernel_size=output_residual_mid_inner_kernel,
                outer_kernel_size=output_residual_outer_kernel,
                detail_scale=output_residual_detail_scale,
                mid_scale=output_residual_mid_scale,
                contrast_scale=output_residual_contrast_scale,
                output_init_std=output_residual_init_std,
            )
        else:
            self.output_residual_refiner = None

    def _limit_high_coeff(self, high: torch.Tensor) -> torch.Tensor:
        if not self.limit_output_high:
            return high
        return self.output_high_scale * torch.tanh(high)

    def reconstruct_single(self, low_feat, high_feat):
        low = self.low_head(low_feat)
        high = self.high_head(high_feat)

        batch_size, _, height, width = high.shape
        high = high.reshape(
            batch_size,
            self.out_channels,
            3,
            height,
            width,
        )
        high = self._limit_high_coeff(high)

        output_logit = self.idwt((low, [high]))
        output_image = self.out_act(output_logit)

        return output_image, low, high, output_logit

    def calibrate_fused(self, fused):
        """Legacy bounded calibration retained for checkpoint compatibility."""
        if not self.use_output_calibration:
            gain = fused.new_ones([])
            bias = fused.new_zeros([])
            return fused, gain, bias

        gain = (
            1.0
            + self.output_calibration_gain_range
            * torch.tanh(self.output_calibration_gain_raw)
        )
        bias = (
            self.output_calibration_bias_range
            * torch.tanh(self.output_calibration_bias_raw)
        )
        center = fused.new_tensor(self.output_calibration_center)
        calibrated = gain * (fused - center) + center + bias
        calibrated = calibrated.clamp(0.0, 1.0)
        return calibrated, gain.squeeze(), bias.squeeze()

    def forward(self, decoder_out, mode="reconstruct"):
        if mode == "reconstruct":
            ir_out, ir_low, ir_high, _ = self.reconstruct_single(
                decoder_out["ir"]["low"],
                decoder_out["ir"]["high"],
            )
            vis_out, vis_low, vis_high, _ = self.reconstruct_single(
                decoder_out["vis"]["low"],
                decoder_out["vis"]["high"],
            )

            return {
                "ir": {
                    "image": ir_out,
                    "low": ir_low,
                    "high": ir_high,
                },
                "vis": {
                    "image": vis_out,
                    "low": vis_low,
                    "high": vis_high,
                },
            }

        if mode == "fusion":
            fused_feature_pack = decoder_out["fused"]
            base_image, fused_low, fused_high, base_logit = (
                self.reconstruct_single(
                    fused_feature_pack["low"],
                    fused_feature_pack["high"],
                )
            )

            if self.output_residual_refiner is not None:
                residual_pack = self.output_residual_refiner(
                    base_logit=base_logit,
                    low_feature=fused_feature_pack["low"],
                    high_feature=fused_feature_pack["high"],
                )
                refined_image = residual_pack["image"]
            else:
                zero = base_image.new_zeros([])
                residual_pack = {
                    "image": base_image,
                    "base_image": base_image.detach(),
                    "base_logit": base_logit.detach(),
                    "refined_logit": base_logit,
                    "residual_raw": torch.zeros(
                        base_image.shape[0],
                        base_image.shape[1] * 3,
                        base_image.shape[2],
                        base_image.shape[3],
                        device=base_image.device,
                        dtype=base_image.dtype,
                    ),
                    "residual_detail": torch.zeros_like(base_image),
                    "residual_mid": torch.zeros_like(base_image),
                    "residual_contrast": torch.zeros_like(base_image),
                    "residual_logit": torch.zeros_like(base_image),
                    "residual_detail_logit": torch.zeros_like(base_image),
                    "residual_mid_logit": torch.zeros_like(base_image),
                    "residual_contrast_logit": torch.zeros_like(base_image),
                    "image_residual": torch.zeros_like(base_image),
                    "image_residual_ratio": zero,
                    "logit_residual_ratio": zero,
                    "detail_logit_ratio": zero,
                    "mid_logit_ratio": zero,
                    "contrast_logit_ratio": zero,
                    "image_residual_abs_mean": zero,
                    "logit_residual_abs_mean": zero,
                    "detail_logit_abs_mean": zero,
                    "mid_logit_abs_mean": zero,
                    "contrast_logit_abs_mean": zero,
                }
                refined_image = base_image

            pre_calibration_image = refined_image
            fused_image, calibration_gain, calibration_bias = (
                self.calibrate_fused(pre_calibration_image)
            )

            return {
                "fused": {
                    "image": fused_image,
                    "base_image": residual_pack["base_image"],
                    "low": fused_low,
                    "high": fused_high,
                    "base_logit": residual_pack["base_logit"],
                    "refined_logit": residual_pack["refined_logit"],
                    "output_residual": residual_pack["image_residual"],
                    "output_residual_logit": residual_pack[
                        "residual_logit"
                    ],
                    "output_residual_detail": residual_pack[
                        "residual_detail_logit"
                    ],
                    "output_residual_mid": residual_pack[
                        "residual_mid_logit"
                    ],
                    "output_residual_contrast": residual_pack[
                        "residual_contrast_logit"
                    ],
                    "output_residual_ratio": residual_pack[
                        "image_residual_ratio"
                    ],
                    "output_residual_logit_ratio": residual_pack[
                        "logit_residual_ratio"
                    ],
                    "output_residual_abs_mean": residual_pack[
                        "image_residual_abs_mean"
                    ],
                    "output_residual_logit_abs_mean": residual_pack[
                        "logit_residual_abs_mean"
                    ],
                    "output_residual_detail_ratio": residual_pack[
                        "detail_logit_ratio"
                    ],
                    "output_residual_mid_ratio": residual_pack[
                        "mid_logit_ratio"
                    ],
                    "output_residual_contrast_ratio": residual_pack[
                        "contrast_logit_ratio"
                    ],
                    "output_residual_detail_abs_mean": residual_pack[
                        "detail_logit_abs_mean"
                    ],
                    "output_residual_mid_abs_mean": residual_pack[
                        "mid_logit_abs_mean"
                    ],
                    "output_residual_contrast_abs_mean": residual_pack[
                        "contrast_logit_abs_mean"
                    ],
                    "pre_calibration_image": pre_calibration_image,
                    "calibration_gain": calibration_gain,
                    "calibration_bias": calibration_bias,
                }
            }

        raise ValueError("mode must be 'reconstruct' or 'fusion'")

