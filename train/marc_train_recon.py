import os
import json
import time
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.marc_fusion_net import MARCFusionNet
from metric.marc_losses import build_material_stage_loss
from utils.loss_visual import AverageMeter, tensor_to_float, save_loss_history_csv, save_loss_history_json ,plot_loss_curves
import matplotlib

matplotlib.use("Agg")


def get_arg(args, name: str, default: Any):
    """
    安全读取 JSON 参数。
    如果 args 中没有该字段，则使用默认值。
    """
    if args is None:
        return default
    return getattr(args, name, default)

def inspect_tensor(name, x):
    with torch.no_grad():
        x_detach = x.detach()

        nan_count = torch.isnan(x_detach).sum().item()
        inf_count = torch.isinf(x_detach).sum().item()

        x_finite = x_detach[torch.isfinite(x_detach)]

        if x_finite.numel() == 0:
            print(f"[{name}] all values are NaN or Inf")
            return

        min_v = x_finite.min().item()
        max_v = x_finite.max().item()
        mean_v = x_finite.mean().item()

        q001 = torch.quantile(x_finite.float(), 0.001).item()
        q999 = torch.quantile(x_finite.float(), 0.999).item()

        low_ratio = (x_detach < 0).float().mean().item()
        high_ratio = (x_detach > 1).float().mean().item()

        print(
            f"[{name}] "
            f"min={min_v:.4f}, max={max_v:.4f}, mean={mean_v:.4f}, "
            f"q0.1%={q001:.4f}, q99.9%={q999:.4f}, "
            f"<0 ratio={low_ratio:.6f}, >1 ratio={high_ratio:.6f}, "
            f"NaN={nan_count}, Inf={inf_count}"
        )


def inspect_high_feature(name, high):

    inspect_tensor(name, high)

    if high.dim() != 4:
        tqdm.write(f"[{name}] high feature is not 4D, skip subband inspection.")
        return

    c = high.shape[1]

    if c % 3 != 0:
        tqdm.write(
            f"[{name}] channel number {c} is not divisible by 3, "
            f"skip subband split."
        )
        return

    h1, h2, h3 = torch.chunk(high, chunks=3, dim=1)

    inspect_tensor(f"{name}_subband_1", h1)
    inspect_tensor(f"{name}_subband_2", h2)
    inspect_tensor(f"{name}_subband_3", h3)

def inspect_saturation(name, x, eps=1e-4):
    with torch.no_grad():
        x = x.detach()
        zero_ratio = (x <= eps).float().mean().item()
        one_ratio = (x >= 1.0 - eps).float().mean().item()

        print(
            f"[{name}] "
            f"near 0 ratio={zero_ratio:.6f}, "
            f"near 1 ratio={one_ratio:.6f}"
        )

def set_random_seed(seed: int = 2026):
    """
    设置随机种子，便于复现实验。
    """
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

class PairedIRVISDataset(Dataset):

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
            neg_patch_size: int = 128,
    ):
        super().__init__()

        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        self.train_list = train_list

        self.image_height = image_height
        self.image_width = image_width

        self.use_random_crop = use_random_crop
        self.train_patch_size = train_patch_size
        self.neg_patch_size = neg_patch_size

        if train_list is not None and str(train_list).strip() and os.path.exists(train_list):
            self.samples = self._load_from_list(train_list)
        else:
            self.samples = self._load_from_dirs()

        if len(self.samples) == 0:
            raise RuntimeError("No paired IR/VIS images found.")

        if len(self.samples) < 2:
            raise RuntimeError(
                "At least two image pairs are required, because negative patches "
                "must be sampled from another image."
            )

    def _resolve_path(self, path_str: str, root: Path) -> Path:
        path = Path(path_str)

        if path.is_absolute():
            return path

        if path.exists():
            return path

        return root / path_str

    def _load_from_list(self, train_list: str) -> List[Tuple[Path, Path]]:
        """
        支持两种 train_list 格式：

        格式 1：
            filename

        表示 IR 和 VIS 文件夹下同名文件。

        格式 2：
            ir_path vis_path

        表示每行给出一对路径。
        """
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

        # 注意：
        # use_random_crop=True 时，不在这里 resize 到 128×128。
        # 只有关闭随机裁剪时，才使用 image_height / image_width 统一尺寸。
        if not self.use_random_crop:
            if self.image_height is not None and self.image_width is not None:
                img = img.resize((self.image_width, self.image_height), Image.BILINEAR)

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return tensor

    def _sample_negative_index(self, idx: int) -> int:
        """
        随机选择另一张图像作为负样本来源。
        """
        neg_idx = random.randint(0, len(self.samples) - 1)

        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.samples) - 1)

        return neg_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_path, vis_path = self.samples[idx]

        ir = self._load_gray_tensor(ir_path)
        vis = self._load_gray_tensor(vis_path)

        # 当前样本：随机裁剪训练 patch
        if self.use_random_crop:
            ir, vis = random_crop_pair(
                ir,
                vis,
                patch_size=self.train_patch_size,
            )

        # 负样本来自另一张图
        neg_idx = self._sample_negative_index(idx)
        neg_ir_path, neg_vis_path = self.samples[neg_idx]

        neg_ir = self._load_gray_tensor(neg_ir_path)
        neg_vis = self._load_gray_tensor(neg_vis_path)

        # 负样本 patch
        neg_ir_patch, neg_vis_patch = random_crop_pair(
            neg_ir,
            neg_vis,
            patch_size=self.neg_patch_size,
        )

        return {
            "ir": ir,
            "vis": vis,
            "image_id": torch.tensor(idx, dtype=torch.long),

            "neg_ir_patch": neg_ir_patch,
            "neg_vis_patch": neg_vis_patch,
            "neg_image_id": torch.tensor(neg_idx, dtype=torch.long),

            "ir_path": str(ir_path),
            "vis_path": str(vis_path),
            "neg_ir_path": str(neg_ir_path),
            "neg_vis_path": str(neg_vis_path),
        }


def random_material_preserving_augment(
    ir: torch.Tensor,
    vis: torch.Tensor,
    noise_std: float = 0.01,
    blur_prob: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造非材质扰动增强视图。

    这些增强改变观测条件，但不改变对应区域的材质属性。

    Args:
        ir, vis:
            [B, 1, H, W]

    Returns:
        ir_aug, vis_aug:
            [B, 1, H, W]
    """
    # -----------------------------
    # VIS: 亮度、对比度、Gamma 扰动
    # -----------------------------
    vis_aug = vis

    brightness = 0.9 + 0.2 * torch.rand(
        vis.shape[0], 1, 1, 1,
        device=vis.device,
        dtype=vis.dtype,
    )

    contrast = 0.9 + 0.2 * torch.rand(
        vis.shape[0], 1, 1, 1,
        device=vis.device,
        dtype=vis.dtype,
    )

    vis_mean = vis_aug.mean(dim=(-2, -1), keepdim=True)
    vis_aug = (vis_aug - vis_mean) * contrast + vis_mean
    vis_aug = vis_aug * brightness

    gamma = 0.8 + 0.4 * torch.rand(
        vis.shape[0], 1, 1, 1,
        device=vis.device,
        dtype=vis.dtype,
    )

    vis_aug = torch.clamp(vis_aug, 1e-4, 1.0)
    vis_aug = vis_aug ** gamma

    # -----------------------------
    # IR: 强度缩放、灰度偏移
    # -----------------------------
    ir_aug = ir

    scale = 0.9 + 0.2 * torch.rand(
        ir.shape[0], 1, 1, 1,
        device=ir.device,
        dtype=ir.dtype,
    )

    shift = -0.05 + 0.1 * torch.rand(
        ir.shape[0], 1, 1, 1,
        device=ir.device,
        dtype=ir.dtype,
    )

    ir_aug = ir_aug * scale + shift

    # -----------------------------
    # 两种模态都加轻微噪声
    # -----------------------------
    if noise_std > 0:
        ir_aug = ir_aug + noise_std * torch.randn_like(ir_aug)
        vis_aug = vis_aug + noise_std * torch.randn_like(vis_aug)

    # -----------------------------
    # 随机轻微模糊
    # -----------------------------
    if random.random() < blur_prob:
        ir_aug = F.avg_pool2d(ir_aug, kernel_size=3, stride=1, padding=1)
        vis_aug = F.avg_pool2d(vis_aug, kernel_size=3, stride=1, padding=1)

    ir_aug = torch.clamp(ir_aug, 0.0, 1.0)
    vis_aug = torch.clamp(vis_aug, 0.0, 1.0)

    return ir_aug, vis_aug


def normalize_map(x: np.ndarray) -> np.ndarray:
    """
    将二维图归一化到 0~255。
    """
    x_min = float(x.min())
    x_max = float(x.max())

    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - x_min) / (x_max - x_min)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)

    return x


def tensor_gray_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    """
    将 [1, H, W] 或 [H, W] tensor 转成 RGB uint8 图。
    """
    x = x.detach().float().cpu()

    if x.dim() == 3:
        x = x[0]

    x = x.clamp(0.0, 1.0).numpy()
    x = (x * 255.0).round().astype(np.uint8)

    rgb = np.stack([x, x, x], axis=-1)

    return rgb


def z_m_to_heatmap(z_m: torch.Tensor, index: int = 0) -> np.ndarray:
    """
    将 z_m 转成热力图。

    z_m:
        [B, C, H, W]

    做法：
        对通道维度计算能量图：
            sqrt(mean(z_m^2, dim=C))
    """
    z = z_m[index].detach().float().cpu()

    z_energy = torch.sqrt(torch.mean(z ** 2, dim=0) + 1e-8)
    z_np = z_energy.numpy()

    z_uint8 = normalize_map(z_np)

    heatmap_bgr = cv2.applyColorMap(z_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb

def feature_to_heatmap(feat: torch.Tensor, index: int = 0) -> np.ndarray:
    """
    将特征图转成热力图。

    feat:
        [B, C, H, W]

    做法：
        对通道维度计算能量图：
            sqrt(mean(feat^2, dim=C))
    """
    f = feat[index].detach().float().cpu()

    energy = torch.sqrt(torch.mean(f ** 2, dim=0) + 1e-8)
    energy_np = energy.numpy()

    energy_uint8 = normalize_map(energy_np)

    heatmap_bgr = cv2.applyColorMap(energy_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def split_z_m(z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 z_m 按通道拆成 z_r 和 z_e。

    z_m:
        [B, C, H, W]

    Returns:
        z_r:
            前一半通道，反射相关材质表达
        z_e:
            后一半通道，热发射相关材质表达
    """
    if z_m.dim() != 4:
        raise ValueError(f"z_m should be [B, C, H, W], got {z_m.shape}.")

    c = z_m.shape[1]

    if c < 2:
        raise ValueError("z_m channel number must be at least 2.")

    if c % 2 != 0:
        # 如果通道数是奇数，前半部分给 z_r，后半部分给 z_e
        c_half = c // 2
        z_r = z_m[:, :c_half, :, :]
        z_e = z_m[:, c_half:, :, :]
    else:
        z_r, z_e = torch.chunk(z_m, chunks=2, dim=1)

    return z_r, z_e

def save_zm_visualization(
    ir: torch.Tensor,
    vis: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    save_dir: str,
    step: int,
    max_items: int = 2,
):
    """
    保存 z_m 材质信息图可视化。

    每个样本保存一张横向拼接图：
        IR | VIS | Recon IR | Recon VIS | z_r heatmap | z_e heatmap

    其中：
        z_r：z_m 前一半通道，偏可见光反射相关材质表达
        z_e：z_m 后一半通道，偏红外热发射相关材质表达
    """
    os.makedirs(save_dir, exist_ok=True)

    z_m = outputs["z_m"]
    recon_ir = outputs["recon_ir"]
    recon_vis = outputs["recon_vis"]

    if z_m is None:
        return

    z_r, z_e = split_z_m(z_m)

    b = min(ir.shape[0], max_items)

    for i in range(b):
        ir_img = tensor_gray_to_rgb_uint8(ir[i])
        vis_img = tensor_gray_to_rgb_uint8(vis[i])
        recon_ir_img = tensor_gray_to_rgb_uint8(recon_ir[i])
        recon_vis_img = tensor_gray_to_rgb_uint8(recon_vis[i])

        zr_img = feature_to_heatmap(z_r, index=i)
        ze_img = feature_to_heatmap(z_e, index=i)

        h, w = ir_img.shape[:2]

        zr_img = cv2.resize(zr_img, (w, h), interpolation=cv2.INTER_LINEAR)
        ze_img = cv2.resize(ze_img, (w, h), interpolation=cv2.INTER_LINEAR)

        gap = np.ones((h, 8, 3), dtype=np.uint8) * 255

        grid = np.concatenate(
            [
                ir_img,
                gap,
                vis_img,
                gap,
                recon_ir_img,
                gap,
                recon_vis_img,
                gap,
                zr_img,
                gap,
                ze_img,
            ],
            axis=1,
        )

        labels = [
            "IR",
            "VIS",
            "Recon IR",
            "Recon VIS",
            "z_r",
            "z_e",
        ]

        x_positions = [
            5,
            1 * (w + 8) + 5,
            2 * (w + 8) + 5,
            3 * (w + 8) + 5,
            4 * (w + 8) + 5,
            5 * (w + 8) + 5,
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

        save_path = os.path.join(save_dir, f"step_{step:08d}_sample_{i}_zr_ze.png")
        cv2.imwrite(save_path, grid_bgr)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    step: int,
    save_path: str,
):
    """
    保存 checkpoint。
    """
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


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    ckpt_path: str,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=False)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", 0)

    return start_epoch, global_step


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    统计总参数量和可训练参数量。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def train_material_stage(args):
    """
    第一阶段：材质拟合 / 材质表征预训练。
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

    neg_patch_size = get_arg(args, "neg_patch_size", 64)

    use_random_crop = get_arg(args, "use_random_crop", True)
    train_patch_size = get_arg(args, "train_patch_size", 256)
    neg_patch_size = get_arg(args, "neg_patch_size", 128)

    dataset = PairedIRVISDataset(
        ir_dir=train_ir_dir,
        vis_dir=train_vis_dir,
        train_list=train_list,
        image_height=train_height,
        image_width=train_width,
        use_random_crop=use_random_crop,
        train_patch_size=train_patch_size,
        neg_patch_size=neg_patch_size,
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
    print(f"Negative patch   : {neg_patch_size}")

    # -----------------------------
    # 3. Model
    # -----------------------------
    model = MARCFusionNet(args=args).to(device)

    model.set_train_stage("pretrain_material")
    model.train()

    total_params, trainable_params = count_trainable_parameters(model)

    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------------
    # 4. Loss and optimizer
    # -----------------------------
    criterion = build_material_stage_loss(args).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=get_arg(args, "lr", 1e-4),
        weight_decay=get_arg(args, "weight_decay", 0.0),
    )

    use_amp = get_arg(args, "use_amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # 5. Resume
    # -----------------------------
    resume_path = get_arg(args, "resume_material_weight_path", "")
    start_epoch = 0
    global_step = 0

    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ckpt_path=resume_path,
            device=device,
        )

        print(f"Resume from {resume_path}")
        print(f"Start epoch: {start_epoch}, global step: {global_step}")

    # -----------------------------
    # 6. Training settings
    # -----------------------------
    epochs = get_arg(args, "material_epochs", 20)

    save_dir = get_arg(args, "material_save_dir", "./checkpoints/material_stage")
    visual_dir = get_arg(args, "material_visual_dir", "./visuals/material_stage")

    log_interval = get_arg(args, "log_interval", 20)
    save_interval = get_arg(args, "save_interval", 1000)
    visualize_interval = get_arg(args, "visualize_interval", 500)
    visualize_max_items = get_arg(args, "visualize_max_items", 2)

    material_loss_type = get_arg(args, "material_loss_type", "negative_patch_infonce")
    material_weight = get_arg(args, "material_consistency_weight", 0.1)
    material_interval = get_arg(args, "material_consistency_interval", 1)

    detach_negative_z = get_arg(args, "detach_negative_z", False)

    grad_clip = get_arg(args, "grad_clip", 0.0)
    loss_log_dir = get_arg(args, "material_loss_log_dir", "./logs/material_stage")
    loss_csv_path = os.path.join(loss_log_dir, "loss_history.csv")
    loss_json_path = os.path.join(loss_log_dir, "loss_history.json")
    loss_curve_path = os.path.join(loss_log_dir, "loss_curve.png")

    os.makedirs(loss_log_dir, exist_ok=True)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print(f"Material loss type : {material_loss_type}")
    print(f"Material weight    : {material_weight}")
    print(f"Material interval  : {material_interval}")
    print(f"Detach negative z  : {detach_negative_z}")
    print(f"Use AMP            : {use_amp}")
    print(f"Grad clip          : {grad_clip}")

    # -----------------------------
    # 7. Training loop
    # -----------------------------
    loss_history = []
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        meter_total = AverageMeter()
        meter_recon = AverageMeter()
        meter_grad = AverageMeter()
        meter_material = AverageMeter()
        meter_decorr = AverageMeter()
        meter_wavelet = AverageMeter()

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=140,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)

            neg_ir_patch = batch["neg_ir_patch"].to(device, non_blocking=True)
            neg_vis_patch = batch["neg_vis_patch"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            do_material_loss = (
                material_weight > 0
                and material_loss_type != "none"
                and material_interval > 0
                and global_step % material_interval == 0
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 当前图像对输出
                outputs = model(
                    ir,
                    vis,
                    stage="pretrain_material",
                    return_aux=False,
                    return_wavelet_detail=True,
                )

                outputs_aug1 = None
                outputs_aug2 = None
                outputs_neg = None

                if do_material_loss:
                    # 同一图像对的两种非材质扰动增强
                    ir_aug_1, vis_aug_1 = random_material_preserving_augment(ir, vis)
                    ir_aug_2, vis_aug_2 = random_material_preserving_augment(ir, vis)

                    outputs_aug1 = model(
                        ir_aug_1,
                        vis_aug_1,
                        stage="pretrain_material",
                        return_aux=False,
                    )

                    outputs_aug2 = model(
                        ir_aug_2,
                        vis_aug_2,
                        stage="pretrain_material",
                        return_aux=False,
                    )

                    # 来自另一张图像对的负样本 patch
                    outputs_neg = model(
                        neg_ir_patch,
                        neg_vis_patch,
                        stage="pretrain_material",
                        return_aux=False,
                    )

                    if detach_negative_z:
                        outputs_neg = dict(outputs_neg)
                        outputs_neg["z_m"] = outputs_neg["z_m"].detach()

                loss, loss_dict = criterion(
                    outputs=outputs,
                    ir=ir,
                    vis=vis,
                    outputs_aug1=outputs_aug1,
                    outputs_aug2=outputs_aug2,
                    outputs_neg=outputs_neg,
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
            loss_recon_value = tensor_to_float(loss_dict["loss_recon"])
            loss_grad_value = tensor_to_float(loss_dict["loss_grad"])
            loss_material_value = tensor_to_float(loss_dict["loss_material"])
            loss_decorr_value = tensor_to_float(loss_dict.get("loss_zr_ze_decorr", 0.0))
            loss_wavelet_value = tensor_to_float(loss_dict.get("loss_wavelet", 0.0))

            meter_total.update(loss_total_value, n=current_batch_size)
            meter_recon.update(loss_recon_value, n=current_batch_size)
            meter_grad.update(loss_grad_value, n=current_batch_size)
            meter_material.update(loss_material_value, n=current_batch_size)
            meter_decorr.update(loss_decorr_value, n=current_batch_size)
            meter_wavelet.update(loss_wavelet_value, n=current_batch_size)

            pbar.set_postfix({
                "loss": f"{float(loss_dict['loss_total']):.4f}",
                "rec": f"{float(loss_dict['loss_recon']):.4f}",
                "grad": f"{float(loss_dict['loss_grad']):.4f}",
                "mat": f"{float(loss_dict['loss_material']):.4f}",
                "decorr": f"{float(loss_dict.get('loss_zr_ze_decorr', 0.0)):.4f}",
                "wav": f"{loss_wavelet_value:.4f}",
                "step": global_step,
            })

            # -----------------------------
            # Log
            # -----------------------------
            if log_interval > 0 and global_step % log_interval == 0:
                log_msg = (
                    f"[Material Stage] "
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Iter [{batch_idx + 1}/{len(loader)}] "
                    f"Step {global_step} "
                    f"Loss: {loss_total_value:.6f} "
                    f"Recon: {loss_recon_value:.6f} "
                    f"Grad: {loss_grad_value:.6f} "
                    f"Material: {loss_material_value:.6f} "
                    f"Decorr: {loss_decorr_value:.6f} "
                    f"| EpochAvg Loss: {meter_total.avg:.6f} "
                    f"Recon: {meter_recon.avg:.6f} "
                    f"Grad: {meter_grad.avg:.6f} "
                    f"Material: {meter_material.avg:.6f} "
                    f"Decorr: {meter_decorr.avg:.6f}"
                )

                # tqdm.write(log_msg)

            # -----------------------------
            # Visualization
            # -----------------------------
            if visualize_interval > 0 and global_step % visualize_interval == 0:
                model.eval()

                with torch.no_grad():
                    vis_outputs = model(
                        ir,
                        vis,
                        stage="pretrain_material",
                        return_aux=False,
                    )

                    save_zm_visualization(
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
                    f"material_step_{global_step:08d}.pth",
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
            "loss_recon": meter_recon.avg,
            "loss_grad": meter_grad.avg,
            "loss_material": meter_material.avg,
            "loss_zr_ze_decorr": meter_decorr.avg,
            "loss_wavelet": meter_wavelet.avg,
            "epoch_time_sec": epoch_time,
        }

        loss_history.append(epoch_record)

        save_loss_history_csv(loss_history, loss_csv_path)
        save_loss_history_json(loss_history, loss_json_path)
        plot_loss_curves(loss_history, loss_curve_path)

        save_path = os.path.join(
            save_dir,
            f"material_epoch_{epoch + 1:03d}.pth",
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
            f"Epoch {epoch + 1} finished. "
            f"Time: {epoch_time:.2f}s. "
            f"AvgLoss: {meter_total.avg:.6f}, "
            f"AvgRecon: {meter_recon.avg:.6f}, "
            f"AvgGrad: {meter_grad.avg:.6f}, "
            f"AvgMaterial: {meter_material.avg:.6f}, "
            f"AvgDecorr: {meter_decorr.avg:.6f}. "
            f"Checkpoint saved: {save_path}. "
            f"Loss curve saved: {loss_curve_path}"
        )

    final_path = os.path.join(save_dir, "material_final.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epochs,
        step=global_step,
        save_path=final_path,
    )

    print(f"Material stage training finished. Final checkpoint: {final_path}")


if __name__ == "__main__":
    json_path = "../params/default/train_material.json"

    args = get_arg(json_path)

    train_material_stage(args)