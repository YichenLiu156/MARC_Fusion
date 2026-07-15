import csv
import matplotlib
import os
import json

from typing import Any, List, Optional, Tuple, Dict



import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

class AverageMeter:
    """
    用于统计一个 epoch 内的平均 loss。
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
    """
    将 tensor 或 float 安全转为 float。
    """
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def save_loss_history_csv(history: List[Dict[str, float]], save_path: str):
    """
    保存 epoch 平均 loss 到 CSV。
    """
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = list(history[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_loss_history_json(history: List[Dict[str, float]], save_path: str):
    """
    保存 epoch 平均 loss 到 JSON。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def plot_loss_curves(history: List[Dict[str, float]], save_path: str):
    """
    绘制并保存 loss 变化曲线。
    """
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [item["epoch"] for item in history]

    loss_total = [item["loss_total"] for item in history]
    loss_recon = [item["loss_recon"] for item in history]
    loss_grad = [item["loss_grad"] for item in history]
    loss_material = [item["loss_material"] for item in history]
    loss_decorr = [item["loss_zr_ze_decorr"] for item in history]

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, loss_total, marker="o", label="total")
    plt.plot(epochs, loss_recon, marker="o", label="recon")
    plt.plot(epochs, loss_grad, marker="o", label="grad")
    plt.plot(epochs, loss_material, marker="o", label="material")
    plt.plot(epochs, loss_decorr, marker="o", label="decorr")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Material Stage Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()