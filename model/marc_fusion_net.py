import os
import json
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn


from model.marc_backbone import FDEncoder, FDDecoder, OutputLayers

from model.material_module import MaterialPriorExtractor
from model.mrc_module import MaterialAwareMRF




def get_arg(args, name: str, default: Any):
    """
    安全读取 args 中的参数。

    如果 args 为 None，或者 JSON 中没有该字段，就返回 default。
    """
    if args is None:
        return default
    return getattr(args, name, default)


class MARCFusionNet(nn.Module):
    """
    Material-aware Reliability-Consistency Fusion Network.

    整体流程：

        IR, VIS
          ↓
        独立或共享 FDEncoder
          ↓
        E_IR^1, E_VIS^1
          ↓
        MaterialPriorExtractor 得到 z_m
          ↓
        DWT + 低频/高频编码
          ↓
        FDDecoder 得到 IR/VIS 的低频和高频解码特征
          ↓
        MRF^L / MRF^H
          ↓
        OutputLayers + IDWT
          ↓
        fused image

    支持两个阶段：

        stage="pretrain_material":
            输出 recon_ir, recon_vis, z_m
            不经过 MRF，不输出 fused

        stage="fusion":
            输出 fused
            可选输出 recon_ir, recon_vis
    """

    def __init__(self, args=None):
        super().__init__()

        self.args = args

        # -----------------------------
        # 1. Basic network parameters
        # -----------------------------
        self.in_channels = get_arg(args, "in_channels", 1)
        self.out_channels = get_arg(args, "out_channels", 1)
        self.base_channels = get_arg(args, "base_channels", 64)
        self.wave = get_arg(args, "wave", "haar")

        self.default_stage = get_arg(args, "default_stage", "fusion")

        # IR / VIS 是否使用不共享的浅层特征提取器
        self.use_independent_stem = get_arg(args, "use_independent_stem", True)

        # 是否默认返回中间特征
        self.return_aux_default = get_arg(args, "return_aux", True)

        # fusion 阶段是否默认返回 recon_ir / recon_vis
        self.return_reconstruction_default = get_arg(args, "return_reconstruction", True)

        # -----------------------------
        # 2. Material module parameters
        # -----------------------------
        self.use_material = get_arg(args, "use_material", True)

        self.material_mode = get_arg(args, "material_mode", "downsampled_ssm")
        self.material_hidden_channels = get_arg(args, "material_hidden_channels", self.base_channels)
        self.material_out_channels = get_arg(args, "material_out_channels", self.base_channels)
        self.material_downsample_ratio = get_arg(args, "material_downsample_ratio", 2)
        self.material_bidirectional = get_arg(args, "material_bidirectional", True)

        # -----------------------------
        # 3. MRF parameters
        # -----------------------------
        self.mrf_hidden_channels = get_arg(args, "mrf_hidden_channels", self.base_channels)

        self.use_low_mrf = get_arg(args, "use_low_mrf", True)
        self.use_high_mrf = get_arg(args, "use_high_mrf", True)
        self.use_low_consistency = get_arg(args, "use_low_consistency", True)
        self.use_high_reliability = get_arg(args, "use_high_reliability", True)

        # -----------------------------
        # 4. Encoder
        # -----------------------------
        if self.use_independent_stem:
            # 红外和可见光不共享 FDEncoder
            self.encoder_ir = FDEncoder(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                wave=self.wave,
            )

            self.encoder_vis = FDEncoder(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                wave=self.wave,
            )
        else:
            # 红外和可见光共享同一个 FDEncoder
            self.encoder_shared = FDEncoder(
                in_channels=self.in_channels,
                base_channels=self.base_channels,
                wave=self.wave,
            )

        # -----------------------------
        # 5. Material prior extractor
        # -----------------------------
        if self.use_material:
            self.material_extractor = MaterialPriorExtractor(
                in_channels=self.base_channels,
                hidden_channels=self.material_hidden_channels,
                out_channels=self.material_out_channels,
                mode=self.material_mode,
                downsample_ratio=self.material_downsample_ratio,
                bidirectional=self.material_bidirectional,
                return_aux=self.return_aux_default,
            )
        else:
            self.material_extractor = None

        # -----------------------------
        # 6. Frequency decoder
        # -----------------------------
        self.decoder = FDDecoder(
            channels=self.base_channels,
        )

        # -----------------------------
        # 7. MRF module
        # -----------------------------
        self.mrf = MaterialAwareMRF(
            feature_channels=self.base_channels,
            material_channels=self.material_out_channels,
            hidden_channels=self.mrf_hidden_channels,
            use_material=self.use_material,
            use_low_mrf=self.use_low_mrf,
            use_high_mrf=self.use_high_mrf,
            use_low_consistency=self.use_low_consistency,
            use_high_reliability=self.use_high_reliability,
            return_aux=self.return_aux_default,
        )

        # -----------------------------
        # 8. Output layer / IDWT
        # -----------------------------
        # self.output_layers = OutputLayers(
        #     channels=self.base_channels,
        #     wave=self.wave,
        #     out_channels=self.out_channels,
        # )
        self.output_layers = OutputLayers(
            channels=self.base_channels,
            wave=self.wave,
            out_channels=self.out_channels,
            output_high_scale=get_arg(args, "output_high_scale", 0.5),
            limit_output_high=get_arg(args, "limit_output_high", True),
            use_output_bn=get_arg(args, "use_output_bn", False),
        )

    # =========================================================
    # Utility functions
    # =========================================================

    def _encode_single(self, encoder: nn.Module, x: torch.Tensor) -> Dict[str, Any]:
        """
        对单一路径进行 FDEncoder 编码。
        """
        stem_feat, low_feats, high_feats, yl, yh = encoder(x)

        return {
            "stem": stem_feat,
            "low_feats": low_feats,
            "high_feats": high_feats,
            "yl": yl,
            "yh": yh,
        }

    def _encode_pair(self, ir: torch.Tensor, vis: torch.Tensor) -> Dict[str, Any]:
        """
        对 IR / VIS 同时编码。
        """
        if self.use_independent_stem:
            ir_pack = self._encode_single(self.encoder_ir, ir)
            vis_pack = self._encode_single(self.encoder_vis, vis)
        else:
            ir_pack = self._encode_single(self.encoder_shared, ir)
            vis_pack = self._encode_single(self.encoder_shared, vis)

        return {
            "ir": ir_pack,
            "vis": vis_pack,
        }

    def _extract_material(
        self,
        e_ir_1: torch.Tensor,
        e_vis_1: torch.Tensor,
        return_aux: bool,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        """
        从 IR/VIS 浅层特征中提取材质先验 z_m。
        """
        if not self.use_material:
            return None, {}

        z_m, material_aux = self.material_extractor(
            e_ir_1,
            e_vis_1,
            return_aux=return_aux,
        )

        return z_m, material_aux

    def _apply_material_intervention(
            self,
            z_m: Optional[torch.Tensor],
            material_intervention: str = "none",
    ) -> Optional[torch.Tensor]:
        """
        材质先验干预函数，用于消融实验。

        material_intervention:
            "none" / "full" / "normal":
                不干预，使用原始 z_m = [z_r, z_e]

            "zero" / "zero_zm" / "wo_material":
                z_m = [0, 0]

            "only_zr":
                z_m = [z_r, 0]

            "only_ze":
                z_m = [0, z_e]

            "swap":
                z_m = [z_e, z_r]
        """
        if z_m is None:
            return None

        if material_intervention is None:
            material_intervention = "none"

        mode = str(material_intervention).lower()

        if mode in {"none", "full", "normal"}:
            return z_m

        if mode in {"zero", "zero_zm", "wo_material", "without_material"}:
            return torch.zeros_like(z_m)

        if z_m.dim() != 4:
            raise ValueError(f"z_m should be [B, C, H, W], got {z_m.shape}")

        c = z_m.shape[1]

        if c % 2 != 0:
            raise ValueError(
                f"z_m channel number must be even for z_r / z_e split, got {c}."
            )

        z_r, z_e = torch.chunk(z_m, chunks=2, dim=1)

        if mode in {"only_zr", "zr", "keep_zr"}:
            z_e_zero = torch.zeros_like(z_e)
            return torch.cat([z_r, z_e_zero], dim=1)

        if mode in {"only_ze", "ze", "keep_ze"}:
            z_r_zero = torch.zeros_like(z_r)
            return torch.cat([z_r_zero, z_e], dim=1)

        if mode in {"swap", "swap_zr_ze", "swap_ze_zr"}:
            return torch.cat([z_e, z_r], dim=1)

        raise ValueError(
            f"Unsupported material_intervention: {material_intervention}. "
            f"Expected 'none', 'zero', 'only_zr', 'only_ze', or 'swap'."
        )

    def _decode_pair(self, encoded: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        对 IR/VIS 的 low/high features 分别解码。

        返回：
            {
                "ir":  {"low": ..., "high": ...},
                "vis": {"low": ..., "high": ...}
            }
        """
        decoder_out = self.decoder(
            low_feats_ir=encoded["ir"]["low_feats"],
            high_feats_ir=encoded["ir"]["high_feats"],
            low_feats_vis=encoded["vis"]["low_feats"],
            high_feats_vis=encoded["vis"]["high_feats"],
            mode="reconstruct",
        )

        return decoder_out

    def _reconstruct_modal_images(
        self,
        decoder_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        使用 OutputLayers 重建 IR / VIS。
        """
        recon_out = self.output_layers(
            decoder_out,
            mode="reconstruct",
        )
        return recon_out

    def _fuse_decoded_features(
        self,
        decoder_out: Dict[str, Dict[str, torch.Tensor]],
        z_m: Optional[torch.Tensor],
        return_aux: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        使用 MRF 融合 decoder 后的 low/high 特征。
        """
        low_ir = decoder_out["ir"]["low"]
        low_vis = decoder_out["vis"]["low"]

        high_ir = decoder_out["ir"]["high"]
        high_vis = decoder_out["vis"]["high"]

        low_fused, high_fused, mrf_aux = self.mrf(
            low_ir=low_ir,
            low_vis=low_vis,
            high_ir=high_ir,
            high_vis=high_vis,
            z_m=z_m,
            return_aux=return_aux,
        )

        return low_fused, high_fused, mrf_aux

    def _output_fused_image(
        self,
        low_fused: torch.Tensor,
        high_fused: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        将融合后的 low/high 特征送入 OutputLayers，并通过 IDWT 输出融合图像。
        """
        decoder_out_fused = {
            "fused": {
                "low": low_fused,
                "high": high_fused,
            }
        }

        fused_out = self.output_layers(
            decoder_out_fused,
            mode="fusion",
        )

        return fused_out

    # =========================================================
    # Stage forward functions
    # =========================================================

    def forward_pretrain_material(
            self,
            ir: torch.Tensor,
            vis: torch.Tensor,
            return_aux: bool,
            return_wavelet_detail: bool = False,
    ) -> Dict[str, Any]:
        """
        第一阶段：材质表征预训练阶段。

        该阶段：
            1. 提取 IR/VIS 浅层特征
            2. 得到 z_m
            3. 分别重建 IR / VIS
            4. 不进行融合，不经过 MRF
        """
        encoded = self._encode_pair(ir, vis)

        e_ir_1 = encoded["ir"]["stem"]
        e_vis_1 = encoded["vis"]["stem"]

        z_m, material_aux = self._extract_material(
            e_ir_1,
            e_vis_1,
            return_aux=return_aux,
        )

        decoder_out = self._decode_pair(encoded)
        recon_out = self._reconstruct_modal_images(decoder_out)

        outputs = {
            "stage": "pretrain_material",

            "recon_ir": recon_out["ir"]["image"],
            "recon_vis": recon_out["vis"]["image"],

            "z_m": z_m,
            "material_aux": material_aux,
        }

        # 只返回 low/high 小波系数细节，用于小波域重建损失。
        # 不额外返回 encoded / decoder_out，避免显存上涨太多。
        if return_wavelet_detail:
            outputs["recon_detail"] = recon_out

        if return_aux:
            outputs.update({
                "encoded": encoded,
                "decoder_out": decoder_out,
                "recon_detail": recon_out,
            })

        return outputs

    def forward_fusion(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        return_aux: bool,
        return_reconstruction: bool,
        material_intervention: str = "none"
    ) -> Dict[str, Any]:
        """
        第二阶段：完整融合训练 / 测试阶段。

        该阶段：
            1. 提取 IR/VIS 浅层特征
            2. 得到 z_m
            3. 分别解码 IR/VIS low/high features
            4. 使用 MRF 融合 low/high features
            5. 使用 OutputLayers + IDWT 得到融合图像
            6. 可选返回 recon_ir / recon_vis
        """
        encoded = self._encode_pair(ir, vis)

        e_ir_1 = encoded["ir"]["stem"]
        e_vis_1 = encoded["vis"]["stem"]

        z_m, material_aux = self._extract_material(
            e_ir_1,
            e_vis_1,
            return_aux=return_aux,
        )

        # 保存原始 z_m，方便后续调试或对比
        z_m_raw = z_m

        # --------------------------------------------------
        # Material intervention for ablation study
        # 实验1：material_intervention="zero" 时，将 z_m 置零
        # --------------------------------------------------
        z_m = self._apply_material_intervention(
            z_m=z_m,
            material_intervention=material_intervention,
        )

        decoder_out = self._decode_pair(encoded)

        low_fused, high_fused, mrf_aux = self._fuse_decoded_features(
            decoder_out=decoder_out,
            z_m=z_m,
            return_aux=return_aux,
        )

        fused_out = self._output_fused_image(
            low_fused=low_fused,
            high_fused=high_fused,
        )

        outputs = {
            "stage": "fusion",

            "fused": fused_out["fused"]["image"],
            "fused_low": fused_out["fused"]["low"],
            "fused_high": fused_out["fused"]["high"],

            # z_m 是干预后的材质先验。
            # 当 material_intervention="zero" 时，这里就是全零。
            "z_m": z_m,

            # z_m_raw 是干预前的原始材质先验，便于调试。
            "z_m_raw": z_m_raw,

            "material_intervention": material_intervention,
            "material_aux": material_aux,
            "mrf_aux": mrf_aux,
        }

        if return_reconstruction:
            recon_out = self._reconstruct_modal_images(decoder_out)

            outputs.update({
                "recon_ir": recon_out["ir"]["image"],
                "recon_vis": recon_out["vis"]["image"],
            })

            if return_aux:
                outputs["recon_detail"] = recon_out

        if return_aux:
            outputs.update({
                "encoded": encoded,
                "decoder_out": decoder_out,
                "low_fused_feat": low_fused,
                "high_fused_feat": high_fused,
                "fused_detail": fused_out,
            })

        return outputs

    # =========================================================
    # Public forward
    # =========================================================

    def forward(
            self,
            ir: torch.Tensor,
            vis: torch.Tensor,
            stage: Optional[str] = None,
            return_aux: Optional[bool] = None,
            return_reconstruction: Optional[bool] = None,
            return_wavelet_detail: bool = False,
            material_intervention: str = "none",
    ) -> Dict[str, Any]:
        """
        Args:
            ir:
                [B, 1, H, W] 红外图像
            vis:
                [B, 1, H, W] 可见光图像
            stage:
                "pretrain_material" 或 "fusion"
            return_aux:
                是否返回中间特征
            return_reconstruction:
                fusion 阶段是否返回 recon_ir / recon_vis

        Returns:
            outputs: dict
        """
        if ir.shape != vis.shape:
            raise ValueError(
                f"ir and vis must have the same shape, got {ir.shape} and {vis.shape}."
            )

        if ir.dim() != 4:
            raise ValueError("ir and vis must be 4D tensors: [B, C, H, W].")

        if stage is None:
            stage = self.default_stage

        if return_aux is None:
            return_aux = self.return_aux_default

        if return_reconstruction is None:
            return_reconstruction = self.return_reconstruction_default

        if stage in {"fusion", "train_fusion", "test"}:
            return self.forward_fusion(
                ir=ir,
                vis=vis,
                return_aux=return_aux,
                return_reconstruction=return_reconstruction,
                material_intervention=material_intervention,
            )

        if stage in {"fusion", "train_fusion", "test"}:
            return self.forward_fusion(
                ir=ir,
                vis=vis,
                return_aux=return_aux,
                return_reconstruction=return_reconstruction,
            )

        raise ValueError(
            f"Unsupported stage: {stage}. "
            f"Expected 'pretrain_material' or 'fusion'."
        )

    # =========================================================
    # Trainable stage control
    # =========================================================

    def set_train_stage(self, stage: str, freeze_material: Optional[bool] = None):
        """
        根据训练阶段设置 requires_grad。

        注意：
            这个函数只是辅助函数。
            如果你的 optimizer 已经创建，修改 requires_grad 后，
            最好重新创建 optimizer 或确保 optimizer 只包含 requires_grad=True 的参数。
        """
        # 先全部打开
        for p in self.parameters():
            p.requires_grad = True

        if stage in {"pretrain", "pretrain_material", "material"}:
            # 第一阶段不训练 MRF
            for p in self.mrf.parameters():
                p.requires_grad = False

        elif stage in {"fusion", "train_fusion"}:
            # 第二阶段默认全部训练
            if freeze_material is None:
                freeze_material = get_arg(self.args, "freeze_material_in_fusion", False)

            if freeze_material and self.material_extractor is not None:
                for p in self.material_extractor.parameters():
                    p.requires_grad = False
        else:
            raise ValueError(
                f"Unsupported training stage: {stage}. "
                f"Expected 'pretrain_material' or 'fusion'."
            )

        return self


if __name__ == "__main__":
    import gc
    import time

    def format_mb(num_bytes):
        return num_bytes / 1024 / 1024

    def print_cuda_memory(title, device):
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
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    # -----------------------------
    # 1. Load args
    # -----------------------------
    json_path = "../params_marc/default/fusion_network.json"

    if os.path.exists(json_path):
        args = get_arg(json_path)
        print(f"Loaded config from: {json_path}")
    else:
        args = None
        print("Config file not found. Use default arguments.")

    # -----------------------------
    # 2. Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # -----------------------------
    # 3. Build model
    # -----------------------------
    model = MARCFusionNet(args=args).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------------
    # 4. Test input
    # -----------------------------
    B = get_arg(args, "test_batch_size", 2)
    C = get_arg(args, "in_channels", 1)
    H = get_arg(args, "test_height", 128)
    W = get_arg(args, "test_width", 128)

    ir = torch.randn(B, C, H, W, device=device)
    vis = torch.randn(B, C, H, W, device=device)

    print_cuda_memory("After model and input initialization", device)

    # -----------------------------
    # 5. Stage 1: pretrain_material test
    # -----------------------------
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        start_time = time.time()

        out_pre = model(
            ir,
            vis,
            stage="pretrain_material",
            return_aux=False,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end_time = time.time()

    print("\n========== Stage 1: pretrain_material ==========")
    print("recon_ir :", tuple(out_pre["recon_ir"].shape))
    print("recon_vis:", tuple(out_pre["recon_vis"].shape))
    print("z_m      :", None if out_pre["z_m"] is None else tuple(out_pre["z_m"].shape))
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Stage 1 inference memory usage", device)

    del out_pre
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------
    # 6. Stage 2: fusion test
    # -----------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        start_time = time.time()

        out_fusion = model(
            ir,
            vis,
            stage="fusion",
            return_aux=False,
            return_reconstruction=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end_time = time.time()

    print("\n========== Stage 2: fusion ==========")
    print("fused    :", tuple(out_fusion["fused"].shape))
    print("recon_ir :", tuple(out_fusion["recon_ir"].shape))
    print("recon_vis:", tuple(out_fusion["recon_vis"].shape))
    print("z_m      :", None if out_fusion["z_m"] is None else tuple(out_fusion["z_m"].shape))
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Stage 2 inference memory usage", device)

    del out_fusion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------
    # 7. Training forward + backward test
    # -----------------------------
    model.train()
    model.set_train_stage("fusion")

    ir_train = torch.randn(B, C, H, W, device=device, requires_grad=True)
    vis_train = torch.randn(B, C, H, W, device=device, requires_grad=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()

    out_train = model(
        ir_train,
        vis_train,
        stage="fusion",
        return_aux=False,
        return_reconstruction=True,
    )

    loss = (
        out_train["fused"].mean()
        + 0.5 * out_train["recon_ir"].mean()
        + 0.5 * out_train["recon_vis"].mean()
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print_cuda_memory("Training forward memory usage", device)

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.time()

    print("\n========== Training Test ==========")
    print("fused:", tuple(out_train["fused"].shape))
    print("loss :", float(loss.detach().cpu()))
    print(f"Forward + backward time: {(end_time - start_time) * 1000:.2f} ms")

    print_cuda_memory("Training forward + backward memory usage", device)

    # -----------------------------
    # 8. Cleanup
    # -----------------------------
    model.zero_grad(set_to_none=True)

    del out_train, loss
    del ir, vis, ir_train, vis_train
    del model

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print_cuda_memory("After cleanup", device)