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






# class OutputLayers(nn.Module):
#     """
#     将 decoder 输出的 low/high 频率特征转换为 IDWT 重建的图像。
#
#     输入：
#         low_feat:  [B, C, H/2, W/2]
#         high_feat: [B, 3C, H/2, W/2]
#
#     输出：
#         image: [B, 1, H, W]
#     """
#
#     def __init__(self, channels=64, wave='haar', out_channels=1):
#         super(OutputLayers, self).__init__()
#
#         self.channels = channels
#         self.out_channels = out_channels
#
#         self.idwt = DWTInverse(wave=wave)
#
#         # low-frequency head
#         self.low_head = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(channels, out_channels, kernel_size=3, padding=0),
#             nn.BatchNorm2d(out_channels),
#         )
#
#         # high-frequency head
#         # high_feat 是 3C 通道，输出应为 3 个子带
#         self.high_head = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(channels * 3, out_channels * 3, kernel_size=3, padding=0),
#             nn.BatchNorm2d(out_channels * 3),
#         )
#
#         self.out_act = nn.Sigmoid()
#
#     def reconstruct_single(self, low_feat, high_feat):
#         """
#         单幅图像重建：IR、VIS、Fused 都走这里。
#         """
#
#         # low: [B, 1, H/2, W/2]
#         low = self.low_head(low_feat)
#
#         # high: [B, 3, H/2, W/2]
#         high = self.high_head(high_feat)
#
#         B, _, H, W = high.shape
#
#         # 转成 pytorch_wavelets IDWT 格式：
#         # [B, 1, 3, H/2, W/2]
#         high = high.view(B, self.out_channels, 3, H, W)
#
#         # IDWT
#         out = self.idwt((low, [high]))
#
#         out = self.out_act(out)
#
#         return out, low, high
#
#     def forward(self, decoder_out, mode="reconstruct"):
#         """
#         mode='reconstruct':
#             decoder_out:
#             {
#                 "ir":  {"low": ..., "high": ...},
#                 "vis": {"low": ..., "high": ...}
#             }
#
#         mode='fusion':
#             decoder_out:
#             {
#                 "fused": {"low": ..., "high": ...}
#             }
#         """
#
#         if mode == "reconstruct":
#             ir_out, ir_low, ir_high = self.reconstruct_single(
#                 decoder_out["ir"]["low"],
#                 decoder_out["ir"]["high"]
#             )
#
#             vis_out, vis_low, vis_high = self.reconstruct_single(
#                 decoder_out["vis"]["low"],
#                 decoder_out["vis"]["high"]
#             )
#
#             return {
#                 "ir": {
#                     "image": ir_out,
#                     "low": ir_low,
#                     "high": ir_high,
#                 },
#                 "vis": {
#                     "image": vis_out,
#                     "low": vis_low,
#                     "high": vis_high,
#                 }
#             }
#
#         elif mode == "fusion":
#             fused_out, fused_low, fused_high = self.reconstruct_single(
#                 decoder_out["fused"]["low"],
#                 decoder_out["fused"]["high"]
#             )
#
#             return {
#                 "fused": {
#                     "image": fused_out,
#                     "low": fused_low,
#                     "high": fused_high,
#                 }
#             }
#
#         else:
#             raise ValueError("mode must be 'reconstruct' or 'fusion'")

class OutputLayers(nn.Module):
    """
    将 decoder 输出的 low/high 频率特征转换为 IDWT 重建的图像。

    输入：
        low_feat:  [B, C, H/2, W/2]
        high_feat: [B, 3C, H/2, W/2]

    输出：
        image: [B, 1, H, W]
    """

    def __init__(
        self,
        channels=64,
        wave='haar',
        out_channels=1,
        output_high_scale=0.5,
        limit_output_high=True,
        use_output_bn=False,
    ):
        super(OutputLayers, self).__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.output_high_scale = output_high_scale
        self.limit_output_high = limit_output_high
        self.use_output_bn = use_output_bn

        self.idwt = DWTInverse(wave=wave)

        # -----------------------------
        # low-frequency head
        # -----------------------------
        low_layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=0),
        ]

        # 不推荐在最终 low 小波系数后使用 BN；
        # 如果你确实想保留，可以通过 use_output_bn=True 打开。
        if use_output_bn:
            low_layers.append(nn.BatchNorm2d(out_channels))

        self.low_head = nn.Sequential(*low_layers)

        # -----------------------------
        # high-frequency head
        # -----------------------------
        # high_feat 是 3C 通道，输出为 out_channels * 3 个高频子带通道。
        high_layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 3, out_channels * 3, kernel_size=3, padding=0),
        ]

        # 不推荐在最终 high 小波系数后使用 BN；
        # 你的黑白格很可能和 high BN + eval running stats 不稳定有关。
        if use_output_bn:
            high_layers.append(nn.BatchNorm2d(out_channels * 3))

        self.high_head = nn.Sequential(*high_layers)

        self.out_act = nn.Sigmoid()

    def _limit_high_coeff(self, high):
        """
        限制 IDWT 前的 high-frequency 小波系数幅值。

        输入：
            high: [B, out_channels, 3, H/2, W/2]

        输出：
            high: [B, out_channels, 3, H/2, W/2]
        """
        if not self.limit_output_high:
            return high

        return self.output_high_scale * torch.tanh(high)

    def reconstruct_single(self, low_feat, high_feat):
        """
        单幅图像重建：IR、VIS、Fused 都走这里。
        """

        # low: [B, out_channels, H/2, W/2]
        low = self.low_head(low_feat)

        # high: [B, out_channels * 3, H/2, W/2]
        high = self.high_head(high_feat)

        B, _, H, W = high.shape

        # 转成 pytorch_wavelets IDWT 格式：
        # [B, out_channels, 3, H/2, W/2]
        high = high.reshape(B, self.out_channels, 3, H, W)

        # 关键修改：
        # 限制 high-frequency coefficients，避免 IDWT 产生黑白格伪影。
        high = self._limit_high_coeff(high)

        # IDWT
        out = self.idwt((low, [high]))

        # 输出图像限制到 [0, 1]
        out = self.out_act(out)

        return out, low, high

    def forward(self, decoder_out, mode="reconstruct"):
        """
        mode='reconstruct':
            decoder_out:
            {
                "ir":  {"low": ..., "high": ...},
                "vis": {"low": ..., "high": ...}
            }

        mode='fusion':
            decoder_out:
            {
                "fused": {"low": ..., "high": ...}
            }
        """

        if mode == "reconstruct":
            ir_out, ir_low, ir_high = self.reconstruct_single(
                decoder_out["ir"]["low"],
                decoder_out["ir"]["high"]
            )

            vis_out, vis_low, vis_high = self.reconstruct_single(
                decoder_out["vis"]["low"],
                decoder_out["vis"]["high"]
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
                }
            }

        elif mode == "fusion":
            fused_out, fused_low, fused_high = self.reconstruct_single(
                decoder_out["fused"]["low"],
                decoder_out["fused"]["high"]
            )

            return {
                "fused": {
                    "image": fused_out,
                    "low": fused_low,
                    "high": fused_high,
                }
            }

        else:
            raise ValueError("mode must be 'reconstruct' or 'fusion'")