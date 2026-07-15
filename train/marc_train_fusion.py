import os
import csv
import json
import time
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from model.marc_fusion_net import MARCFusionNet
from metric.marc_losses import build_fusion_stage_loss
from utils.loss_visual import AverageMeter, tensor_to_float, save_loss_history_csv, save_loss_history_json , plot_loss_curves


matplotlib.use("Agg")

def get_arg(args, name: str, default: Any):
    if args is None:
        return default
    return getattr(args, name, default)


def set_random_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_to_min_size(x: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    """
    如果图像小于指定尺寸，则用 replicate padding 补到至少 min_h × min_w。

    x:
        [1, H, W]
    """
    if x.dim() != 3:
        raise ValueError(f"x should be [1, H, W], got {x.shape}")

    _, h, w = x.shape

    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)

    if pad_h == 0 and pad_w == 0:
        return x

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(
        x.unsqueeze(0),
        pad=(pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    ).squeeze(0)

    return x


def random_crop_pair(
    ir: torch.Tensor,
    vis: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对配准 IR / VIS 图像裁剪同一位置 patch。

    输入：
        ir : [1, H, W]
        vis: [1, H, W]

    输出：
        ir_patch : [1, patch_size, patch_size]
        vis_patch: [1, patch_size, patch_size]
    """
    if patch_size % 2 != 0:
        raise ValueError("patch_size should be even because DWT requires even spatial size.")

    if ir.dim() != 3 or vis.dim() != 3:
        raise ValueError(f"ir/vis should be [1, H, W], got {ir.shape}, {vis.shape}")

    # 如果尺寸不一致，先把 VIS 对齐到 IR 尺寸
    if ir.shape[-2:] != vis.shape[-2:]:
        vis = F.interpolate(
            vis.unsqueeze(0),
            size=ir.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    ir = pad_to_min_size(ir, patch_size, patch_size)
    vis = pad_to_min_size(vis, patch_size, patch_size)

    _, h, w = ir.shape

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    ir_patch = ir[:, top:top + patch_size, left:left + patch_size]
    vis_patch = vis[:, top:top + patch_size, left:left + patch_size]

    return ir_patch, vis_patch

def plot_low_vis_bg_diagnostics(history, save_path):
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [item["epoch"] for item in history]

    keys = [
        "loss_low_vis_bg_mask_mean",
        "loss_low_vis_bg_fused_ratio_ir",
    ]

    plt.figure(figsize=(10, 5))

    for key in keys:
        if key in history[0]:
            values = [item[key] for item in history]
            plt.plot(epochs, values, marker="o", label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Low VIS-background Diagnostics")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()
class PairedIRVISDataset(Dataset):
    """
    红外-可见光配准图像对数据集。

    当前版本默认将 IR / VIS 都读成灰度图：

        IR  : [1, H, W]
        VIS : [1, H, W]

    支持两种读取方式：

    1. 不使用 train_list：
        按 ir_dir 和 vis_dir 中排序后的图像一一配对。

    2. 使用 train_list：
        每行一个文件名：
            0001.png

        或每行两个路径：
            ir/0001.png vis/0001.png
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
            self,
            ir_dir: str,
            vis_dir: str,
            train_list: Optional[str] = None,
            image_height: Optional[int] = None,
            image_width: Optional[int] = None,
            use_random_crop: bool = True,
            train_patch_size: int = 256,
    ):
        super().__init__()

        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        self.train_list = train_list

        self.image_height = image_height
        self.image_width = image_width

        self.use_random_crop = use_random_crop
        self.train_patch_size = train_patch_size

        if train_list is not None and str(train_list).strip() and os.path.exists(train_list):
            self.samples = self._load_from_list(train_list)
        else:
            self.samples = self._load_from_dirs()

        if len(self.samples) == 0:
            raise RuntimeError("No paired IR/VIS images found.")

    def _load_from_list(self, train_list: str) -> List[Tuple[Path, Path]]:
        samples = []

        with open(train_list, "r", encoding="utf-8") as f:
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

        if not self.use_random_crop:
            if self.image_height is not None and self.image_width is not None:
                img = img.resize((self.image_width, self.image_height), Image.BILINEAR)

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_path, vis_path = self.samples[idx]

        ir = self._load_gray_tensor(ir_path)
        vis = self._load_gray_tensor(vis_path)

        if self.use_random_crop:
            ir, vis = random_crop_pair(
                ir,
                vis,
                patch_size=self.train_patch_size,
            )

        return {
            "ir": ir,
            "vis": vis,
            "image_id": torch.tensor(idx, dtype=torch.long),
            "ir_path": str(ir_path),
            "vis_path": str(vis_path),
        }


class AverageMeter:
    """
    统计 epoch average loss。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        value = float(value)
        self.sum += value * n
        self.count += n

        if self.count > 0:
            self.avg = self.sum / self.count


def tensor_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def tensor_gray_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    """
    [1, H, W] 或 [H, W] tensor -> RGB uint8。
    """
    x = x.detach().float().cpu()

    if x.dim() == 3:
        x = x[0]

    x = x.clamp(0.0, 1.0).numpy()
    x = (x * 255.0).round().astype(np.uint8)

    rgb = np.stack([x, x, x], axis=-1)

    return rgb

def normalize_map(x: np.ndarray) -> np.ndarray:
    """
    将二维响应图归一化到 0~255。
    """
    x_min = float(x.min())
    x_max = float(x.max())

    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - x_min) / (x_max - x_min)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)

    return x


def split_z_m(z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 z_m 拆成 z_r 和 z_e。

    z_r：前一半通道
    z_e：后一半通道
    """
    if z_m.dim() != 4:
        raise ValueError(f"z_m should be [B, C, H, W], got {z_m.shape}.")

    c = z_m.shape[1]

    if c % 2 != 0:
        c_half = c // 2
        z_r = z_m[:, :c_half, :, :]
        z_e = z_m[:, c_half:, :, :]
    else:
        z_r, z_e = torch.chunk(z_m, chunks=2, dim=1)

    return z_r, z_e


def feature_to_heatmap(feat: torch.Tensor, index: int = 0) -> np.ndarray:
    """
    将特征图转成热力图。

    feat:
        [B, C, H, W]

    方法：
        sqrt(mean(feat^2, dim=C))
    """
    f = feat[index].detach().float().cpu()

    if f.dim() == 3:
        energy = torch.sqrt(torch.mean(f ** 2, dim=0) + 1e-8)
    elif f.dim() == 2:
        energy = f
    else:
        raise ValueError(f"Unsupported feature shape: {f.shape}")

    energy_np = energy.numpy()
    energy_uint8 = normalize_map(energy_np)

    heatmap_bgr = cv2.applyColorMap(energy_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def get_low_consistency_map(outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    从 outputs 中取低频一致性因子 C。

    期望位置：
        outputs["mrf_aux"]["low"]["low_consistency"]

    返回：
        C: [B, C, H, W]
    """
    mrf_aux = outputs.get("mrf_aux", None)

    if not mrf_aux:
        return None

    low_aux = mrf_aux.get("low", None)

    if not low_aux:
        return None

    return low_aux.get("low_consistency", None)


def get_high_reliability_map(outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    从 outputs 中取高频可靠性响应 r。

    HighFrequencyMRF 中通常有：
        high_reliability_ir
        high_reliability_vis

    这里合成为一个 r 图：
        r = max(r_ir, r_vis)

    返回：
        r: [B, C, H, W]
    """
    mrf_aux = outputs.get("mrf_aux", None)

    if not mrf_aux:
        return None

    high_aux = mrf_aux.get("high", None)

    if not high_aux:
        return None

    r_ir = high_aux.get("high_reliability_ir", None)
    r_vis = high_aux.get("high_reliability_vis", None)

    if r_ir is None and r_vis is None:
        return None

    if r_ir is None:
        return r_vis

    if r_vis is None:
        return r_ir

    r = torch.max(r_ir, r_vis)

    return r

def get_high_reliability_maps(
    outputs: Dict[str, torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从 outputs 中分别取高频可靠性图 Qir 和 Qvis。

    期望位置：
        outputs["mrf_aux"]["high"]["high_reliability_ir"]
        outputs["mrf_aux"]["high"]["high_reliability_vis"]

    Returns:
        q_ir:
            红外高频可靠性图

        q_vis:
            可见光高频可靠性图
    """
    mrf_aux = outputs.get("mrf_aux", None)

    if not mrf_aux:
        return None, None

    high_aux = mrf_aux.get("high", None)

    if not high_aux:
        return None, None

    q_ir = high_aux.get("high_reliability_ir", None)
    q_vis = high_aux.get("high_reliability_vis", None)

    return q_ir, q_vis

def save_fusion_visualization(
    ir: torch.Tensor,
    vis: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    save_dir: str,
    step: int,
    max_items: int = 2,
):
    """
    保存融合阶段可视化图。

    显示：
        IR | VIS | Fused | z_r | z_e | C | Qir | Qvis | Recon IR | Recon VIS

    其中：
        z_r:
            反射偏向材质响应

        z_e:
            热发射偏向材质响应

        C:
            低频一致性因子 low_consistency

        Qir:
            红外高频可靠性图 high_reliability_ir

        Qvis:
            可见光高频可靠性图 high_reliability_vis
    """
    os.makedirs(save_dir, exist_ok=True)

    fused = outputs["fused"]

    has_recon = ("recon_ir" in outputs) and ("recon_vis" in outputs)

    # -----------------------------
    # 1. 材质响应 z_r / z_e
    # -----------------------------
    z_m = outputs.get("z_m", None)

    if z_m is not None:
        z_r, z_e = split_z_m(z_m)
    else:
        z_r, z_e = None, None

    # -----------------------------
    # 2. MRF 中的 C、Qir、Qvis
    # -----------------------------
    c_map = get_low_consistency_map(outputs)
    q_ir_map, q_vis_map = get_high_reliability_maps(outputs)

    b = min(ir.shape[0], max_items)

    for i in range(b):
        ir_img = tensor_gray_to_rgb_uint8(ir[i])
        vis_img = tensor_gray_to_rgb_uint8(vis[i])
        fused_img = tensor_gray_to_rgb_uint8(fused[i])

        h, w = ir_img.shape[:2]
        gap = np.ones((h, 8, 3), dtype=np.uint8) * 255

        img_list = [
            ir_img,
            gap,
            vis_img,
            gap,
            fused_img,
        ]

        labels = ["IR", "VIS", "Fused"]

        # -----------------------------
        # z_r
        # -----------------------------
        if z_r is not None:
            zr_img = feature_to_heatmap(z_r, index=i)
            zr_img = cv2.resize(zr_img, (w, h), interpolation=cv2.INTER_LINEAR)

            img_list.extend([gap, zr_img])
            labels.append("z_r")

        # -----------------------------
        # z_e
        # -----------------------------
        if z_e is not None:
            ze_img = feature_to_heatmap(z_e, index=i)
            ze_img = cv2.resize(ze_img, (w, h), interpolation=cv2.INTER_LINEAR)

            img_list.extend([gap, ze_img])
            labels.append("z_e")

        # -----------------------------
        # C: low consistency
        # -----------------------------
        if c_map is not None:
            c_img = feature_to_heatmap(c_map, index=i)
            c_img = cv2.resize(c_img, (w, h), interpolation=cv2.INTER_LINEAR)

            img_list.extend([gap, c_img])
            labels.append("C")

        # -----------------------------
        # Qir: IR high-frequency reliability
        # -----------------------------
        if q_ir_map is not None:
            qir_img = feature_to_heatmap(q_ir_map, index=i)
            qir_img = cv2.resize(qir_img, (w, h), interpolation=cv2.INTER_LINEAR)

            img_list.extend([gap, qir_img])
            labels.append("Qir")

        # -----------------------------
        # Qvis: VIS high-frequency reliability
        # -----------------------------
        if q_vis_map is not None:
            qvis_img = feature_to_heatmap(q_vis_map, index=i)
            qvis_img = cv2.resize(qvis_img, (w, h), interpolation=cv2.INTER_LINEAR)

            img_list.extend([gap, qvis_img])
            labels.append("Qvis")

        # -----------------------------
        # Recon IR / Recon VIS
        # -----------------------------
        if has_recon:
            recon_ir_img = tensor_gray_to_rgb_uint8(outputs["recon_ir"][i])
            recon_vis_img = tensor_gray_to_rgb_uint8(outputs["recon_vis"][i])

            img_list.extend([
                gap,
                recon_ir_img,
                gap,
                recon_vis_img,
            ])

            labels.extend(["Recon IR", "Recon VIS"])

        grid = np.concatenate(img_list, axis=1)

        x_positions = [
            j * (w + 8) + 5
            for j in range(len(labels))
        ]

        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        for label, x in zip(labels, x_positions):
            cv2.putText(
                grid_bgr,
                label,
                (x, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                grid_bgr,
                label,
                (x, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        save_path = os.path.join(
            save_dir,
            f"step_{step:08d}_sample_{i}_fusion_material_qir_qvis.png"
        )

        cv2.imwrite(save_path, grid_bgr)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    step: int,
    save_path: str,
):
    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    torch.save(ckpt, save_path)


def load_training_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    ckpt_path: str,
    device: torch.device,
    load_optimizer: bool = False,
):
    """
    用于恢复融合阶段训练。

    说明：
    - 默认只加载 model 权重；
    - 不加载旧 optimizer，避免 freeze_material_in_fusion 改变后参数组不匹配；
    - 如果确实需要完整恢复训练状态，可设置 load_optimizer=True。
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1. 加载模型权重
    if "model" in ckpt:
        model_state = ckpt["model"]
    elif "state_dict" in ckpt:
        model_state = ckpt["state_dict"]
    else:
        model_state = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(
        model_state,
        strict=False
    )

    print(f"Loaded fusion checkpoint from: {ckpt_path}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    if len(missing_keys) > 0:
        print("First missing keys:", missing_keys[:10])

    if len(unexpected_keys) > 0:
        print("First unexpected keys:", unexpected_keys[:10])

    # 2. 是否加载 optimizer
    if load_optimizer and optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Optimizer state loaded.")
        except ValueError as e:
            print("[Warning] Optimizer state is incompatible with current trainable parameters.")
            print("[Warning] Skip loading optimizer state.")
            print(f"[Warning] Reason: {e}")
    else:
        print("Skip loading optimizer state.")

    # 3. 是否加载 scaler
    if load_optimizer and scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print("AMP scaler state loaded.")
        except Exception as e:
            print("[Warning] Skip loading scaler state.")
            print(f"[Warning] Reason: {e}")
    else:
        print("Skip loading scaler state.")

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", ckpt.get("global_step", 0))

    return start_epoch, global_step

def load_pretrained_model_weight(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
):
    """
    用于加载材质阶段预训练权重。

    兼容两种 checkpoint 格式：
        1. {"model": state_dict, ...}
        2. 直接 state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded pretrained weight from: {ckpt_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_loss_history_csv(history: List[Dict[str, float]], save_path: str):
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = list(history[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_loss_history_json(history: List[Dict[str, float]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def plot_loss_curves(history: List[Dict[str, float]], save_path: str):
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(10, 6))

    loss_keys = [
        "loss_total",
        "loss_intensity",
        "loss_gradient",
        "loss_ssim",
        "loss_reconstruction",
        "loss_tv",
        "loss_saturation"
        "loss_low_vis_bg",
        "loss_low_vis_bg_fused_w",
        "loss_low_vis_bg_raw_w",
    ]

    for key in loss_keys:
        if key in history[0]:
            values = [item[key] for item in history]
            plt.plot(epochs, values, marker="o", label=key.replace("loss_", ""))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fusion Stage Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()


def train_fusion_stage(args):
    """
    第二阶段：融合训练。
    """
    # -----------------------------
    # 1. Basic settings
    # -----------------------------
    seed = get_arg(args, "seed", 2026)
    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    # -----------------------------
    # 2. Dataset
    # -----------------------------
    train_ir_dir = get_arg(args, "train_ir_dir", "")
    train_vis_dir = get_arg(args, "train_vis_dir", "")
    train_list = get_arg(args, "train_list", "")

    train_height = get_arg(args, "train_height", None)
    train_width = get_arg(args, "train_width", None)

    use_random_crop = get_arg(args, "use_random_crop", True)
    train_patch_size = get_arg(args, "train_patch_size", 256)

    dataset = PairedIRVISDataset(
        ir_dir=train_ir_dir,
        vis_dir=train_vis_dir,
        train_list=train_list,
        image_height=train_height,
        image_width=train_width,
        use_random_crop=use_random_crop,
        train_patch_size=train_patch_size,
    )

    batch_size = get_arg(args, "batch_size", 2)
    num_workers = get_arg(args, "num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"Training samples : {len(dataset)}")
    print(f"Batch size       : {batch_size}")

    # -----------------------------
    # 3. Model
    # -----------------------------
    model = MARCFusionNet(args=args).to(device)

    # 加载材质阶段预训练权重
    pretrained_material_path = get_arg(args, "pretrained_material_weight_path", "")

    if pretrained_material_path and os.path.exists(pretrained_material_path):
        load_pretrained_model_weight(
            model=model,
            ckpt_path=pretrained_material_path,
            device=device,
            strict=False,
        )
    else:
        print("[Warning] No pretrained material checkpoint loaded.")

    freeze_material = get_arg(args, "freeze_material_in_fusion", False)

    model.set_train_stage(
        stage="fusion",
        freeze_material=freeze_material,
    )

    model.train()

    total_params, trainable_params = count_trainable_parameters(model)

    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Freeze material     : {freeze_material}")

    # -----------------------------
    # 4. Loss and optimizer
    # -----------------------------
    criterion = build_fusion_stage_loss(args).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=get_arg(args, "lr", 5e-5),
        weight_decay=get_arg(args, "weight_decay", 0.0),
    )

    use_amp = get_arg(args, "use_amp", False) and device.type == "cuda"

    if use_amp:
        print("[Warning] AMP may be incompatible with pytorch_wavelets. If dtype error occurs, set use_amp=false.")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # 5. Resume fusion training
    # -----------------------------
    resume_path = get_arg(args, "resume_fusion_weight_path", "")
    start_epoch = 0
    global_step = 0

    if resume_path and os.path.exists(resume_path):
        resume_load_optimizer = getattr(args, "resume_load_optimizer", False)

        start_epoch, global_step = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ckpt_path=resume_path,
            device=device,
            load_optimizer=resume_load_optimizer,
        )

        print(f"Resume fusion training from {resume_path}")
        print(f"Start epoch: {start_epoch}, global step: {global_step}")

    # -----------------------------
    # 6. Training settings
    # -----------------------------
    epochs = get_arg(args, "fusion_epochs", 20)

    save_dir = get_arg(args, "fusion_save_dir", "./checkpoints/fusion_stage")
    visual_dir = get_arg(args, "fusion_visual_dir", "./visuals/fusion_stage")
    loss_log_dir = get_arg(args, "fusion_loss_log_dir", "./logs/fusion_stage")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(loss_log_dir, exist_ok=True)

    loss_csv_path = os.path.join(loss_log_dir, "loss_history.csv")
    loss_json_path = os.path.join(loss_log_dir, "loss_history.json")
    loss_curve_path = os.path.join(loss_log_dir, "loss_curve.png")
    low_vis_diag_path = os.path.join(loss_log_dir, "low_vis_bg_diagnostics.png")

    log_interval = get_arg(args, "log_interval", 20)
    save_interval = get_arg(args, "save_interval", 1000)
    visualize_interval = get_arg(args, "visualize_interval", 500)
    visualize_max_items = get_arg(args, "visualize_max_items", 2)

    grad_clip = get_arg(args, "grad_clip", 0.0)

    # fusion 阶段是否返回 recon_ir / recon_vis
    # 如果 fusion_reconstruction_weight > 0，必须返回 recon。
    fusion_reconstruction_weight = get_arg(args, "fusion_reconstruction_weight", 0.0)
    return_reconstruction = (
        fusion_reconstruction_weight > 0
        or get_arg(args, "return_reconstruction", True)
    )

    # 如果你的 FusionStageLoss 里需要 mrf_aux，可以把 fusion_return_aux 设为 true。
    return_aux = get_arg(args, "fusion_return_aux", False)

    print(f"Fusion epochs       : {epochs}")
    print(f"Learning rate       : {get_arg(args, 'lr', 5e-5)}")
    print(f"Use AMP             : {use_amp}")
    print(f"Grad clip           : {grad_clip}")
    print(f"Return recon        : {return_reconstruction}")
    print(f"Return aux          : {return_aux}")

    loss_history = []

    # -----------------------------
    # 7. Training loop
    # -----------------------------
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        meter_total = AverageMeter()
        meter_intensity = AverageMeter()
        meter_gradient = AverageMeter()
        meter_ssim = AverageMeter()
        meter_reconstruction = AverageMeter()
        meter_tv = AverageMeter()
        meter_saturation = AverageMeter()
        meter_low_vis_bg = AverageMeter()
        meter_low_vis_bg_fused = AverageMeter()
        meter_low_vis_bg_raw = AverageMeter()
        meter_low_vis_bg_mask = AverageMeter()
        meter_low_vis_bg_ratio_ir = AverageMeter()
        meter_low_vis_bg = AverageMeter()
        meter_low_vis_bg_fused = AverageMeter()
        meter_low_vis_bg_raw = AverageMeter()
        meter_low_vis_bg_mask = AverageMeter()
        meter_low_vis_bg_ratio_ir = AverageMeter()

        pbar = tqdm(
            loader,
            desc=f"Fusion Epoch {epoch + 1}/{epochs}",
            ncols=160,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    ir,
                    vis,
                    stage="fusion",
                    return_aux=return_aux,
                    return_reconstruction=return_reconstruction,
                )

                loss, loss_dict = criterion(
                    outputs=outputs,
                    ir=ir,
                    vis=vis,
                )

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            current_batch_size = ir.shape[0]

            loss_total_value = tensor_to_float(loss_dict["loss_total"])
            loss_intensity_value = tensor_to_float(loss_dict.get("loss_intensity", 0.0))
            loss_gradient_value = tensor_to_float(loss_dict.get("loss_gradient", 0.0))
            loss_ssim_value = tensor_to_float(loss_dict.get("loss_ssim", 0.0))
            loss_reconstruction_value = tensor_to_float(loss_dict.get("loss_reconstruction", 0.0))
            loss_tv_value = tensor_to_float(loss_dict.get("loss_tv", 0.0))
            loss_saturation_value = tensor_to_float(loss_dict.get("loss_saturation", 0.0))
            loss_low_vis_bg_value = tensor_to_float(loss_dict.get("loss_low_vis_bg", 0.0))
            loss_low_vis_bg_fused_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_fused_w", 0.0))
            loss_low_vis_bg_raw_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_raw_w", 0.0))
            loss_low_vis_bg_mask_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_mask_mean", 0.0))
            loss_low_vis_bg_ratio_ir_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_fused_ratio_ir", 0.0))
            loss_low_vis_bg_value = tensor_to_float(loss_dict.get("loss_low_vis_bg", 0.0))
            loss_low_vis_bg_fused_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_fused_w", 0.0))
            loss_low_vis_bg_raw_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_raw_w", 0.0))
            loss_low_vis_bg_mask_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_mask_mean", 0.0))
            loss_low_vis_bg_ratio_ir_value = tensor_to_float(loss_dict.get("loss_low_vis_bg_fused_ratio_ir", 0.0))

            meter_total.update(loss_total_value, n=current_batch_size)
            meter_intensity.update(loss_intensity_value, n=current_batch_size)
            meter_gradient.update(loss_gradient_value, n=current_batch_size)
            meter_ssim.update(loss_ssim_value, n=current_batch_size)
            meter_reconstruction.update(loss_reconstruction_value, n=current_batch_size)
            meter_tv.update(loss_tv_value, n=current_batch_size)
            meter_saturation.update(loss_saturation_value, n=current_batch_size)
            meter_low_vis_bg.update(loss_low_vis_bg_value, n=current_batch_size)
            meter_low_vis_bg_fused.update(loss_low_vis_bg_fused_value, n=current_batch_size)
            meter_low_vis_bg_raw.update(loss_low_vis_bg_raw_value, n=current_batch_size)
            meter_low_vis_bg_mask.update(loss_low_vis_bg_mask_value, n=current_batch_size)
            meter_low_vis_bg_ratio_ir.update(loss_low_vis_bg_ratio_ir_value, n=current_batch_size)
            meter_low_vis_bg.update(loss_low_vis_bg_value, n=current_batch_size)
            meter_low_vis_bg_fused.update(loss_low_vis_bg_fused_value, n=current_batch_size)
            meter_low_vis_bg_raw.update(loss_low_vis_bg_raw_value, n=current_batch_size)
            meter_low_vis_bg_mask.update(loss_low_vis_bg_mask_value, n=current_batch_size)
            meter_low_vis_bg_ratio_ir.update(loss_low_vis_bg_ratio_ir_value, n=current_batch_size)

            pbar.set_postfix({
                "loss": f"{loss_total_value:.4f}",
                "avg": f"{meter_total.avg:.4f}",
                "int": f"{loss_intensity_value:.4f}",
                "grad": f"{loss_gradient_value:.4f}",
                "ssim": f"{loss_ssim_value:.4f}",
                "rec": f"{loss_reconstruction_value:.4f}",
                "tv": f"{loss_tv_value:.4f}",
                "step": global_step,
                "sat": f"{loss_saturation_value:.4f}",
                "lowvis": f"{loss_low_vis_bg_value:.4f}",
                "mask": f"{loss_low_vis_bg_mask_value:.3f}",
                "irR": f"{loss_low_vis_bg_ratio_ir_value:.3f}",
            })

            # -----------------------------
            # Log
            # -----------------------------
            if log_interval > 0 and global_step % log_interval == 0:
                log_msg = (
                    f"[Fusion Stage] "
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Iter [{batch_idx + 1}/{len(loader)}] "
                    f"Step {global_step} "
                    f"Loss: {loss_total_value:.6f} "
                    f"Intensity: {loss_intensity_value:.6f} "
                    f"Grad: {loss_gradient_value:.6f} "
                    f"SSIM: {loss_ssim_value:.6f} "
                    f"Recon: {loss_reconstruction_value:.6f} "
                    f"TV: {loss_tv_value:.6f} "
                    f"| EpochAvg Loss: {meter_total.avg:.6f} "
                    f"Intensity: {meter_intensity.avg:.6f} "
                    f"Grad: {meter_gradient.avg:.6f} "
                    f"SSIM: {meter_ssim.avg:.6f} "
                    f"Recon: {meter_reconstruction.avg:.6f} "
                    f"TV: {meter_tv.avg:.6f}"
                )

                tqdm.write(log_msg)

            # -----------------------------
            # Visualization
            # -----------------------------
            if visualize_interval > 0 and global_step % visualize_interval == 0:
                model.eval()

                with torch.no_grad():
                    vis_outputs = model(
                        ir,
                        vis,
                        stage="fusion",
                        return_aux=True,
                        return_reconstruction=True,
                    )

                    save_fusion_visualization(
                        ir=ir,
                        vis=vis,
                        outputs=vis_outputs,
                        save_dir=visual_dir,
                        step=global_step,
                        max_items=visualize_max_items,
                    )

                model.train()

            # -----------------------------
            # Save by step
            # -----------------------------
            if save_interval > 0 and global_step % save_interval == 0:
                save_path = os.path.join(
                    save_dir,
                    f"fusion_step_{global_step:08d}.pth",
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    save_path=save_path,
                )

                tqdm.write(f"Saved checkpoint: {save_path}")

        # -----------------------------
        # Save by epoch
        # -----------------------------
        epoch_time = time.time() - epoch_start_time

        epoch_record = {
            "epoch": epoch + 1,
            "step": global_step,
            "loss_total": meter_total.avg,
            "loss_intensity": meter_intensity.avg,
            "loss_gradient": meter_gradient.avg,
            "loss_ssim": meter_ssim.avg,
            "loss_reconstruction": meter_reconstruction.avg,
            "loss_tv": meter_tv.avg,
            "epoch_time_sec": epoch_time,
            "loss_saturation": meter_saturation.avg,
            "loss_low_vis_bg": meter_low_vis_bg.avg,
            "loss_low_vis_bg_fused_w": meter_low_vis_bg_fused.avg,
            "loss_low_vis_bg_raw_w": meter_low_vis_bg_raw.avg,
            "loss_low_vis_bg_mask_mean": meter_low_vis_bg_mask.avg,
            "loss_low_vis_bg_fused_ratio_ir": meter_low_vis_bg_ratio_ir.avg,

        }

        loss_history.append(epoch_record)

        save_loss_history_csv(loss_history, loss_csv_path)
        save_loss_history_json(loss_history, loss_json_path)
        plot_loss_curves(loss_history, loss_curve_path)

        plot_low_vis_bg_diagnostics(loss_history, low_vis_diag_path)

        save_path = os.path.join(
            save_dir,
            f"fusion_epoch_{epoch + 1:03d}.pth",
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            step=global_step,
            save_path=save_path,
        )

        tqdm.write(
            f"Fusion epoch {epoch + 1} finished. "
            f"Time: {epoch_time:.2f}s. "
            f"AvgLoss: {meter_total.avg:.6f}, "
            f"AvgIntensity: {meter_intensity.avg:.6f}, "
            f"AvgGrad: {meter_gradient.avg:.6f}, "
            f"AvgSSIM: {meter_ssim.avg:.6f}, "
            f"AvgRecon: {meter_reconstruction.avg:.6f}, "
            f"AvgTV: {meter_tv.avg:.6f}. "
            f"Checkpoint saved: {save_path}. "
            f"Loss curve saved: {loss_curve_path}"
        )

    final_path = os.path.join(save_dir, "fusion_final.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epochs,
        step=global_step,
        save_path=final_path,
    )

    print(f"Fusion stage training finished. Final checkpoint: {final_path}")


if __name__ == "__main__":
    json_path = "../params_marc/default/train_fusion.json"

    args = get_arg(json_path)

    train_fusion_stage(args)