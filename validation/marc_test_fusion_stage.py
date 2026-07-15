import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.marc_fusion_net import MARCFusionNet


FEATURE_KEYS = ["zr", "ze", "c", "qir", "qvis"]


class GetArgs:
    """
    支持测试配置 + 网络配置合并。

    fusion_test.json 中可以写：
        "network_config_path": "../params_marc/default/fusion_network.json"

    最终 args 同时包含：
        测试参数 + 网络结构参数

    如果两个 json 中有同名字段：
        fusion_test.json 中的字段优先级更高。
    """

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            test_args = json.load(f)

        network_config_path = test_args.get("network_config_path", "")

        merged_args = {}

        if network_config_path:
            with open(network_config_path, "r", encoding="utf-8") as f:
                network_args = json.load(f)

            merged_args.update(network_args)

        merged_args.update(test_args)

        self.__dict__.update(merged_args)


def get_arg(args, name: str, default: Any):
    if args is None:
        return default
    return getattr(args, name, default)


class PairedIRVISTestDataset(Dataset):
    """
    测试集读取。

    推荐目录结构：

        test/
        ├── ir/
        │   ├── 0001.png
        │   ├── 0002.png
        │   └── ...
        └── vis/
            ├── 0001.png
            ├── 0002.png
            └── ...

    默认读取为灰度：
        IR  : [1, H, W]
        VIS : [1, H, W]
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        ir_dir: str,
        vis_dir: str,
        test_list: Optional[str] = None,
    ):
        super().__init__()

        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        self.test_list = test_list

        if test_list is not None and str(test_list).strip() and os.path.exists(test_list):
            self.samples = self._load_from_list(test_list)
        else:
            self.samples = self._load_from_dirs()

        if len(self.samples) == 0:
            raise RuntimeError("No paired IR/VIS test images found.")

    def _resolve_path(self, path_str: str, root: Path) -> Path:
        path = Path(path_str)

        if path.is_absolute():
            return path

        if path.exists():
            return path

        return root / path_str

    def _load_from_list(self, test_list: str) -> List[Tuple[Path, Path]]:
        """
        支持两种 test_list 格式：

        格式 1：
            0001.png

        表示：
            test_ir_dir/0001.png
            test_vis_dir/0001.png

        格式 2：
            ir_path vis_path
        """
        samples = []

        with open(test_list, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) == 1:
                filename = parts[0]
                ir_path = self.ir_dir / filename
                vis_path = self.vis_dir / filename
            else:
                ir_path = self._resolve_path(parts[0], self.ir_dir)
                vis_path = self._resolve_path(parts[1], self.vis_dir)

            if ir_path.exists() and vis_path.exists():
                samples.append((ir_path, vis_path))

        return samples

    def _load_from_dirs(self) -> List[Tuple[Path, Path]]:
        ir_files = sorted([
            p for p in self.ir_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        ])

        vis_files = sorted([
            p for p in self.vis_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        ])

        if len(ir_files) != len(vis_files):
            print(
                f"[Warning] IR image number {len(ir_files)} != "
                f"VIS image number {len(vis_files)}. "
                f"Use minimum paired length."
            )

        min_len = min(len(ir_files), len(vis_files))

        return list(zip(ir_files[:min_len], vis_files[:min_len]))

    def _load_gray_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_path, vis_path = self.samples[idx]

        ir = self._load_gray_tensor(ir_path)
        vis = self._load_gray_tensor(vis_path)

        # 如果 IR / VIS 尺寸不同，将 VIS 对齐到 IR 尺寸。
        if ir.shape[-2:] != vis.shape[-2:]:
            vis = F.interpolate(
                vis.unsqueeze(0),
                size=ir.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        name = ir_path.stem

        return {
            "ir": ir,
            "vis": vis,
            "name": name,
            "ir_path": str(ir_path),
            "vis_path": str(vis_path),
        }


def load_model_weight(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
):
    """
    加载融合阶段权重。

    支持两种格式：
        1. {"model": state_dict, ...}
        2. 直接 state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded weight from: {ckpt_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])


def pad_to_patch_size(
    x: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将输入 padding 到至少 patch_size。

    x:
        [1, 1, H, W]

    返回：
        padded_x
        original_size: (H, W)
    """
    if x.dim() != 4:
        raise ValueError(f"x should be [B, C, H, W], got {x.shape}")

    _, _, h, w = x.shape
    original_size = (h, w)

    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    if pad_h > 0 or pad_w > 0:
        pad_mode = "reflect" if h > 1 and w > 1 else "replicate"

        x = F.pad(
            x,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode=pad_mode,
        )

    return x, original_size


def get_start_positions(length: int, patch_size: int, stride: int) -> List[int]:
    """
    生成滑窗起点，保证最后一个 patch 覆盖到图像边界。
    """
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))

    last = length - patch_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def build_blend_weight(
    patch_size: int,
    device: torch.device,
    use_hanning: bool = True,
) -> torch.Tensor:
    """
    构造 patch 拼接权重。

    use_hanning=True:
        使用 Hann 窗，减少拼接边界。

    use_hanning=False:
        使用全 1 权重，重叠区域普通平均。
    """
    if not use_hanning:
        return torch.ones(1, 1, patch_size, patch_size, device=device)

    win_1d = torch.hann_window(
        patch_size,
        periodic=False,
        device=device,
    )

    weight = torch.outer(win_1d, win_1d)

    # Hann 窗边缘为 0，裁剪拼接时容易导致边缘权重过低，这里设置下限。
    weight = weight.clamp_min(1e-3)

    weight = weight.view(1, 1, patch_size, patch_size)

    return weight


def split_z_m(z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 z_m 拆成 z_r 和 z_e。

    z_m:
        [B, C, H, W]

    z_r:
        前一半通道，反射偏向材质响应。

    z_e:
        后一半通道，热发射偏向材质响应。
    """
    if z_m.dim() != 4:
        raise ValueError(f"z_m should be [B, C, H, W], got {z_m.shape}")

    c = z_m.shape[1]

    if c < 2:
        raise ValueError("z_m channel number must be at least 2.")

    if c % 2 == 0:
        z_r, z_e = torch.chunk(z_m, chunks=2, dim=1)
    else:
        c_half = c // 2
        z_r = z_m[:, :c_half, :, :]
        z_e = z_m[:, c_half:, :, :]

    return z_r, z_e


def feature_to_response_tensor(
    feat: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """
    将特征图转换成单通道响应图。

    支持输入：
        [B, C, H, W]
        [B, C, 3, H, W]

    输出：
        [B, 1, target_size, target_size]
    """
    if feat.dim() == 5:
        B, C, S, H, W = feat.shape
        feat = feat.reshape(B, C * S, H, W)

    if feat.dim() != 4:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    response = torch.sqrt(torch.mean(feat ** 2, dim=1, keepdim=True) + 1e-8)

    if response.shape[-2:] != (target_size, target_size):
        response = F.interpolate(
            response,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    return response


def extract_patch_feature_maps(
    outputs: Dict[str, torch.Tensor],
    patch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    从单个 patch 的模型输出中提取：
        z_r, z_e, C, Qir, Qvis

    返回：
        {
            "zr":   [1, 1, patch_size, patch_size],
            "ze":   [1, 1, patch_size, patch_size],
            "c":    [1, 1, patch_size, patch_size],
            "qir":  [1, 1, patch_size, patch_size],
            "qvis": [1, 1, patch_size, patch_size],
        }
    """
    feature_maps = {}

    # -----------------------------
    # z_r / z_e
    # -----------------------------
    z_m = outputs.get("z_m", None)

    if z_m is not None:
        z_r, z_e = split_z_m(z_m)

        feature_maps["zr"] = feature_to_response_tensor(
            z_r,
            target_size=patch_size,
        )

        feature_maps["ze"] = feature_to_response_tensor(
            z_e,
            target_size=patch_size,
        )

    # -----------------------------
    # C / Qir / Qvis
    # -----------------------------
    mrf_aux = outputs.get("mrf_aux", None)

    if mrf_aux is not None:
        low_aux = mrf_aux.get("low", None)
        high_aux = mrf_aux.get("high", None)

        if low_aux is not None:
            c_map = low_aux.get("low_consistency", None)

            if c_map is not None:
                feature_maps["c"] = feature_to_response_tensor(
                    c_map,
                    target_size=patch_size,
                )

        if high_aux is not None:
            qir = high_aux.get("high_reliability_ir", None)
            qvis = high_aux.get("high_reliability_vis", None)

            if qir is not None:
                feature_maps["qir"] = feature_to_response_tensor(
                    qir,
                    target_size=patch_size,
                )

            if qvis is not None:
                feature_maps["qvis"] = feature_to_response_tensor(
                    qvis,
                    target_size=patch_size,
                )

    return feature_maps


@torch.no_grad()
@torch.no_grad()
def tiled_fusion_inference(
    model: torch.nn.Module,
    ir: torch.Tensor,
    vis: torch.Tensor,
    patch_size: int = 256,
    stride: int = 128,
    use_hanning: bool = True,
    device: torch.device = torch.device("cuda"),
    return_feature_maps: bool = False,
    tile_border: int = 32,
    material_intervention: str = "none",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    滑窗裁剪推理并拼接。

    改进点：
        1. 每个 patch 推理后，不再使用完整 patch；
        2. 丢掉 patch 四周 tile_border 像素；
        3. 只拼接中间稳定区域；
        4. 图像边界处保留外侧区域，避免输出缺口。

    输入：
        ir : [1, 1, H, W]
        vis: [1, 1, H, W]

    输出：
        fused:
            [1, 1, H, W]

        full_feature_maps:
            {
                "zr":   [1, 1, H, W],
                "ze":   [1, 1, H, W],
                "c":    [1, 1, H, W],
                "qir":  [1, 1, H, W],
                "qvis": [1, 1, H, W],
            }
    """
    if patch_size % 2 != 0:
        raise ValueError("patch_size should be even because DWT/IDWT requires even spatial size.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    if tile_border < 0:
        raise ValueError("tile_border must be >= 0.")

    if tile_border * 2 >= patch_size:
        raise ValueError("tile_border is too large. It should be smaller than patch_size / 2.")

    if tile_border > 0 and stride > patch_size - 2 * tile_border:
        raise ValueError(
            "stride is too large for the selected tile_border. "
            "Please make sure: stride <= patch_size - 2 * tile_border. "
            f"Current stride={stride}, patch_size={patch_size}, tile_border={tile_border}."
        )

    if ir.shape != vis.shape:
        raise ValueError(f"ir and vis should have same shape, got {ir.shape}, {vis.shape}")

    ir = ir.to(device)
    vis = vis.to(device)

    ir, original_size = pad_to_patch_size(ir, patch_size)
    vis, _ = pad_to_patch_size(vis, patch_size)

    _, _, h, w = ir.shape

    h_positions = get_start_positions(h, patch_size, stride)
    w_positions = get_start_positions(w, patch_size, stride)

    output_accum = torch.zeros(1, 1, h, w, device=device)
    weight_accum = torch.zeros(1, 1, h, w, device=device)

    blend_weight = build_blend_weight(
        patch_size=patch_size,
        device=device,
        use_hanning=use_hanning,
    )

    if return_feature_maps:
        feature_accum = {
            key: torch.zeros(1, 1, h, w, device=device)
            for key in FEATURE_KEYS
        }
        feature_available = {
            key: False
            for key in FEATURE_KEYS
        }
    else:
        feature_accum = {}
        feature_available = {}

    for top in h_positions:
        for left in w_positions:
            ir_patch = ir[:, :, top:top + patch_size, left:left + patch_size]
            vis_patch = vis[:, :, top:top + patch_size, left:left + patch_size]


            outputs = model(
                ir_patch,
                vis_patch,
                stage="fusion",
                return_aux=return_feature_maps,
                return_reconstruction=False,
                material_intervention=material_intervention,
            )

            # 统计值
            # from MARCFusion.utils.mrf_raw_stats import (
            #     compute_raw_mrf_stats,
            #     print_mrf_stats,
            #     save_mrf_stats_csv,
            # )
            # from MARCFusion.utils.low_mrf_bias_stats import (
            #     compute_low_mrf_branch_bias_stats,
            #     print_low_mrf_branch_bias_stats,
            #     save_low_mrf_branch_bias_csv,
            # )
            # rows = compute_raw_mrf_stats(
            #     outputs=outputs,
            #     ir=ir,
            #     vis=vis,
            #     sample_name="test_sample_001",
            # )
            #
            # print_mrf_stats(rows)
            #
            # save_mrf_stats_csv(
            #     rows,
            #     "./mrf_raw_stats_test_sample_001.csv",
            # )

            # rows = compute_low_mrf_branch_bias_stats(
            #     outputs=outputs,
            #     ir=ir,
            #     vis=vis,
            #     sample_name="test_sample_001",
            # )
            #
            # print_low_mrf_branch_bias_stats(rows)
            #
            # save_low_mrf_branch_bias_csv(
            #     rows,
            #     "./low_mrf_branch_bias_test_sample_001.csv",
            # )

            fused_patch = outputs["fused"]

            # -------------------------------------------------
            # 关键：计算当前 patch 的有效区域
            # 图像边界处不裁掉外侧，否则会有空洞
            # -------------------------------------------------
            y0 = 0 if top == 0 else tile_border
            x0 = 0 if left == 0 else tile_border

            y1 = patch_size if top + patch_size >= h else patch_size - tile_border
            x1 = patch_size if left + patch_size >= w else patch_size - tile_border

            global_y0 = top + y0
            global_y1 = top + y1
            global_x0 = left + x0
            global_x1 = left + x1

            fused_valid = fused_patch[:, :, y0:y1, x0:x1]
            weight_valid = blend_weight[:, :, y0:y1, x0:x1]

            output_accum[:, :, global_y0:global_y1, global_x0:global_x1] += (
                fused_valid * weight_valid
            )

            weight_accum[:, :, global_y0:global_y1, global_x0:global_x1] += weight_valid

            # -------------------------------------------------
            # 特征图也使用同样的有效区域拼接
            # -------------------------------------------------
            if return_feature_maps:
                patch_feature_maps = extract_patch_feature_maps(
                    outputs=outputs,
                    patch_size=patch_size,
                )

                for key, fmap in patch_feature_maps.items():
                    if key not in feature_accum:
                        continue

                    fmap_valid = fmap[:, :, y0:y1, x0:x1]

                    feature_accum[key][:, :, global_y0:global_y1, global_x0:global_x1] += (
                        fmap_valid * weight_valid
                    )

                    feature_available[key] = True

    fused = output_accum / weight_accum.clamp_min(1e-8)

    ori_h, ori_w = original_size
    fused = fused[:, :, :ori_h, :ori_w]
    fused = fused.clamp(0.0, 1.0)

    full_feature_maps = {}

    if return_feature_maps:
        for key in FEATURE_KEYS:
            if not feature_available[key]:
                continue

            fmap = feature_accum[key] / weight_accum.clamp_min(1e-8)
            fmap = fmap[:, :, :ori_h, :ori_w]
            full_feature_maps[key] = fmap

    return fused, full_feature_maps


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """
    将 tensor 转为 uint8 灰度图。

    支持：
        [1, 1, H, W]
        [1, H, W]
        [H, W]
    """
    x = x.detach().float().cpu()

    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]
    elif x.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    x = x.clamp(0.0, 1.0).numpy()
    x = (x * 255.0).round().astype(np.uint8)

    return x


def save_gray_image(x: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = tensor_to_uint8_image(x)
    cv2.imwrite(save_path, img)


def gray_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    img = tensor_to_uint8_image(x)
    return np.stack([img, img, img], axis=-1)


def normalize_map_to_uint8(x: np.ndarray) -> np.ndarray:
    """
    单通道响应图归一化到 0~255。
    """
    x_min = float(x.min())
    x_max = float(x.max())

    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - x_min) / (x_max - x_min)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)

    return x


def save_feature_heatmap(
    feature_map: torch.Tensor,
    save_path: str,
):
    """
    保存单通道特征响应图为彩色 heatmap。

    feature_map:
        [1, 1, H, W]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = feature_map.detach().float().cpu()[0, 0].numpy()
    x_uint8 = normalize_map_to_uint8(x)

    heatmap_bgr = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap_bgr)


def save_feature_npy(
    feature_map: torch.Tensor,
    save_path: str,
):
    """
    保存原始单通道响应图为 .npy。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = feature_map.detach().float().cpu()[0, 0].numpy()
    np.save(save_path, x)

def inspect_fused_saturation(name: str, fused: torch.Tensor):
    """
    检查融合图是否存在过亮/饱和问题。

    fused:
        [1, 1, H, W]
    """
    with torch.no_grad():
        x = fused.detach().float()

        print(
            f"\n[Fused Stats] {name}\n"
            f"min        : {x.min().item():.6f}\n"
            f"max        : {x.max().item():.6f}\n"
            f"mean       : {x.mean().item():.6f}\n"
            f"std        : {x.std().item():.6f}\n"
            f">0.80 ratio: {(x > 0.80).float().mean().item():.6f}\n"
            f">0.85 ratio: {(x > 0.85).float().mean().item():.6f}\n"
            f">0.90 ratio: {(x > 0.90).float().mean().item():.6f}\n"
            f">0.95 ratio: {(x > 0.95).float().mean().item():.6f}\n"
            f">0.98 ratio: {(x > 0.98).float().mean().item():.6f}\n"
            f"near1 ratio: {(x >= 1.0 - 1e-4).float().mean().item():.6f}\n"
        )


def inspect_feature_map_stats(name: str, feature_map: torch.Tensor):
    """
    检查 C / Qir / Qvis / zr / ze 的真实数值分布。

    feature_map:
        [1, 1, H, W]
    """
    with torch.no_grad():
        x = feature_map.detach().float()

        print(
            f"\n[Feature Map Stats] {name}\n"
            f"min        : {x.min().item():.6f}\n"
            f"max        : {x.max().item():.6f}\n"
            f"mean       : {x.mean().item():.6f}\n"
            f"std        : {x.std().item():.6f}\n"
            f">0.30 ratio: {(x > 0.30).float().mean().item():.6f}\n"
            f">0.50 ratio: {(x > 0.50).float().mean().item():.6f}\n"
            f">0.70 ratio: {(x > 0.70).float().mean().item():.6f}\n"
            f">0.90 ratio: {(x > 0.90).float().mean().item():.6f}\n"
        )

def save_test_visualization(
    ir: torch.Tensor,
    vis: torch.Tensor,
    fused: torch.Tensor,
    save_path: str,
):
    """
    保存测试可视化：
        IR | VIS | Fused
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ir_img = gray_to_rgb_uint8(ir)
    vis_img = gray_to_rgb_uint8(vis)
    fused_img = gray_to_rgb_uint8(fused)

    h, w = ir_img.shape[:2]
    gap = np.ones((h, 8, 3), dtype=np.uint8) * 255

    grid = np.concatenate(
        [
            ir_img,
            gap,
            vis_img,
            gap,
            fused_img,
        ],
        axis=1,
    )

    labels = ["IR", "VIS", "Fused"]

    x_positions = [
        5,
        w + 8 + 5,
        2 * (w + 8) + 5,
    ]

    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    for label, x in zip(labels, x_positions):
        cv2.putText(
            grid_bgr,
            label,
            (x, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            grid_bgr,
            label,
            (x, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, grid_bgr)


def test_fusion_stage(args):
    """
    融合模型测试。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    # -----------------------------
    # 1. Dataset
    # -----------------------------
    test_ir_dir = get_arg(args, "test_ir_dir", "")
    test_vis_dir = get_arg(args, "test_vis_dir", "")
    test_list = get_arg(args, "test_list", "")

    dataset = PairedIRVISTestDataset(
        ir_dir=test_ir_dir,
        vis_dir=test_vis_dir,
        test_list=test_list,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test samples: {len(dataset)}")

    # -----------------------------
    # 2. Model
    # -----------------------------
    model = MARCFusionNet(args=args).to(device)

    fusion_weight_path = get_arg(args, "fusion_weight_path", "")

    if not fusion_weight_path or not os.path.exists(fusion_weight_path):
        raise FileNotFoundError(f"fusion_weight_path not found: {fusion_weight_path}")

    load_model_weight(
        model=model,
        ckpt_path=fusion_weight_path,
        device=device,
        strict=False,
    )

    if hasattr(model, "set_train_stage"):
        model.set_train_stage(
            stage="fusion",
            freeze_material=False,
        )

    model.eval()

    # -----------------------------
    # 3. Test settings
    # -----------------------------
    patch_size = get_arg(args, "test_patch_size", 256)
    stride = get_arg(args, "test_stride", patch_size // 2)
    use_hanning = get_arg(args, "test_use_hanning", True)

    material_intervention = get_arg(args, "test_material_intervention", "none")

    fused_save_dir = get_arg(args, "test_fused_save_dir", "./test_results/fused")
    visual_save_dir = get_arg(args, "test_visual_save_dir", "./test_results/visual")

    save_feature_maps = get_arg(args, "test_save_feature_maps", False)
    save_feature_npy_flag = get_arg(args, "test_save_feature_npy", False)

    print_fused_stats = get_arg(args, "test_print_fused_stats", False)
    print_feature_stats = get_arg(args, "test_print_feature_stats", False)
    max_print_samples = get_arg(args, "test_print_stats_samples", 5)

    zr_save_dir = get_arg(args, "test_zr_save_dir", "./test_results/zr")
    ze_save_dir = get_arg(args, "test_ze_save_dir", "./test_results/ze")
    c_save_dir = get_arg(args, "test_c_save_dir", "./test_results/c")
    qir_save_dir = get_arg(args, "test_qir_save_dir", "./test_results/qir")
    qvis_save_dir = get_arg(args, "test_qvis_save_dir", "./test_results/qvis")

    feature_save_dirs = {
        "zr": zr_save_dir,
        "ze": ze_save_dir,
        "c": c_save_dir,
        "qir": qir_save_dir,
        "qvis": qvis_save_dir,
    }

    os.makedirs(fused_save_dir, exist_ok=True)
    os.makedirs(visual_save_dir, exist_ok=True)

    if save_feature_maps:
        for d in feature_save_dirs.values():
            os.makedirs(d, exist_ok=True)

    print(f"Patch size       : {patch_size}")
    print(f"Stride           : {stride}")
    print(f"Hanning          : {use_hanning}")
    print(f"Save fused       : {fused_save_dir}")
    print(f"Save visual      : {visual_save_dir}")
    print(f"Save feature maps: {save_feature_maps}")
    print(f"Print fused stats  : {print_fused_stats}")
    print(f"Print feature stats: {print_feature_stats}")
    print(f"Print stat samples : {max_print_samples}")
    print(f"Material intervention: {material_intervention}")

    if save_feature_maps:
        print(f"Save z_r         : {zr_save_dir}")
        print(f"Save z_e         : {ze_save_dir}")
        print(f"Save C           : {c_save_dir}")
        print(f"Save Qir         : {qir_save_dir}")
        print(f"Save Qvis        : {qvis_save_dir}")
        print(f"Save npy         : {save_feature_npy_flag}")

    # -----------------------------
    # 4. Inference
    # -----------------------------
    for batch_idx, batch in enumerate(tqdm(loader, desc="Testing", ncols=120)):
        ir = batch["ir"]
        vis = batch["vis"]
        name = batch["name"][0]

        tile_border = get_arg(args, "test_tile_border", 32)

        fused, feature_maps = tiled_fusion_inference(
            model=model,
            ir=ir,
            vis=vis,
            patch_size=patch_size,
            stride=stride,
            use_hanning=use_hanning,
            device=device,
            return_feature_maps=save_feature_maps,
            tile_border=tile_border,
            material_intervention=material_intervention,
        )

        # -----------------------------
        # Optional stats printing
        # -----------------------------
        if batch_idx < max_print_samples:
            if print_fused_stats:
                inspect_fused_saturation(name, fused)

            if print_feature_stats and save_feature_maps:
                if "c" in feature_maps:
                    inspect_feature_map_stats(f"{name} - C", feature_maps["c"])

                if "qir" in feature_maps:
                    inspect_feature_map_stats(f"{name} - Qir", feature_maps["qir"])

                if "qvis" in feature_maps:
                    inspect_feature_map_stats(f"{name} - Qvis", feature_maps["qvis"])

                if "zr" in feature_maps:
                    inspect_feature_map_stats(f"{name} - z_r", feature_maps["zr"])

                if "ze" in feature_maps:
                    inspect_feature_map_stats(f"{name} - z_e", feature_maps["ze"])

        fused_path = os.path.join(fused_save_dir, f"{name}.png")
        visual_path = os.path.join(visual_save_dir, f"{name}_visual.png")

        save_gray_image(fused, fused_path)
        save_test_visualization(ir, vis, fused, visual_path)

        if save_feature_maps:
            for key, fmap in feature_maps.items():
                if key not in feature_save_dirs:
                    continue

                heatmap_path = os.path.join(
                    feature_save_dirs[key],
                    f"{name}_{key}.png",
                )

                save_feature_heatmap(fmap, heatmap_path)

                if save_feature_npy_flag:
                    npy_path = os.path.join(
                        feature_save_dirs[key],
                        f"{name}_{key}.npy",
                    )

                    save_feature_npy(fmap, npy_path)

    print("Testing finished.")


if __name__ == "__main__":
    # json_path = "../params_marc/wo_mrf_h_fusion_test.json"
    json_path = "../params/default/fusion_test.json"
    args = GetArgs(json_path)

    test_fusion_stage(args)