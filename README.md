# MARC-Fusion

This repository contains the PyTorch implementation associated with the paper:

> **Material-aware Selective Fusion for Infrared-Visible Images**

MARC-Fusion learns material-related latent priors and uses them to guide frequency-decoupled infrared-visible image fusion. The training procedure contains two stages:

1. **Material-prior pretraining**
2. **Fusion-network training**

The repository also provides tiled inference for testing images at their original resolution.

## Project Structure

A recommended project layout is:

```text
MARC-Fusion/
├── model/
│   └── marc_fusion_net.py
├── metric/
│   └── marc_losses.py
├── utils/
├── params/
│   └── default/
│       ├── fusion_network.json
│       ├── train_material.json
│       ├── train_fusion.json
│       └── fusion_test.json
├── datasets/
│   ├── train/
│   │   ├── ir/
│   │   └── vi/
│   └── test/
│       ├── ir/
│       └── vi/
├── weights/
│   └── fusion_weights.pth
├── train_material.py
├── train_fusion.py
├── test_fusion.py
└── README.md
```

The directory names are configurable. Update the corresponding paths in the JSON files when a different layout is used.

## Environment

Create a Python environment and install the required packages:

```bash
pip install torch torchvision
pip install numpy pillow opencv-python tqdm matplotlib pytorch-wavelets
```

Alternatively, install all dependencies from the repository:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for training.

## Dataset Preparation

Each dataset or split must contain two folders:

```text
dataset_root/
├── ir/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── vi/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

- `ir/` contains infrared images.
- `vi/` contains visible images.
- Infrared and visible images must be spatially registered.
- Corresponding image pairs should use the same filename.
- The numbers of infrared and visible images should be equal.
- Supported formats are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.
- Images are loaded as single-channel images. RGB visible images are converted to grayscale by the current data loader.
- Training patch sizes must be even because the network uses DWT/IDWT operations.
- Material-stage training requires at least two paired samples because negative patches are sampled from another image pair.

### Recommended training and testing layout

```text
datasets/
├── train/
│   ├── ir/
│   └── vi/
└── test/
    ├── ir/
    └── vi/
```

Set the paths in the configuration files as follows:

```json
{
  "train_ir_dir": "./datasets/train/ir",
  "train_vis_dir": "./datasets/train/vi"
}
```

```json
{
  "test_ir_dir": "./datasets/test/ir",
  "test_vis_dir": "./datasets/test/vi"
}
```

### Optional file list

When `train_list` or `test_list` is empty, files in the infrared and visible folders are sorted and paired in order.

A list file can also be used. Two formats are supported.

One shared filename per line:

```text
0001.png
0002.png
0003.png
```

Or one infrared-visible path pair per line:

```text
ir/0001.png vi/0001.png
ir/0002.png vi/0002.png
```

## Pretrained Weights

The pretrained fusion weight can be downloaded from:

> **[Download pretrained weights](WEIGHT_DOWNLOAD_LINK)**

Place the downloaded file at:

```text
weights/
└── fusion_weights.pth
```

The test configuration should contain:

```json
{
  "fusion_weight_path": "./weights/fusion_weights.pth"
}
```

The test code supports either of the following checkpoint formats:

```python
{"model": model.state_dict()}
```

or a directly saved `state_dict`.

## Configuration

The scripts use JSON configuration files. Before running a script, check the `json_path` variable near the bottom of that script and ensure that it points to the correct configuration file.

Recommended configuration files:

```text
params/default/train_material.json
params/default/train_fusion.json
params/default/fusion_test.json
params/default/fusion_network.json
```

Paths inside JSON files are resolved relative to the directory from which the Python process is started. Using paths relative to the repository root is recommended.

## Training

### Stage 1: Material-Prior Pretraining

Edit `params/default/train_material.json`.

Important fields include:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "train_ir_dir": "./datasets/train/ir",
  "train_vis_dir": "./datasets/train/vi",
  "train_list": "",
  "material_epochs": 30,
  "batch_size": 1,
  "train_patch_size": 160,
  "neg_patch_size": 128,
  "material_save_dir": "./outputs/material/checkpoints",
  "material_visual_dir": "./outputs/material/visuals",
  "material_loss_log_dir": "./outputs/material/logs",
  "resume_material_weight_path": ""
}
```

Run:

```bash
python train_material.py
```

The material-stage checkpoints are saved to the directory specified by `material_save_dir`. The final checkpoint is saved as:

```text
material_final.pth
```

To resume material training, set:

```json
{
  "resume_material_weight_path": "./outputs/material/checkpoints/material_epoch_020.pth"
}
```

### Stage 2: Fusion-Network Training

After material pretraining, edit `params/default/train_fusion.json`.

Set `pretrained_material_weight_path` to the material-stage checkpoint:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "train_ir_dir": "./datasets/train/ir",
  "train_vis_dir": "./datasets/train/vi",
  "pretrained_material_weight_path": "./outputs/material/checkpoints/material_final.pth",
  "resume_fusion_weight_path": "",
  "fusion_epochs": 50,
  "batch_size": 2,
  "train_patch_size": 128,
  "fusion_save_dir": "./outputs/fusion/checkpoints",
  "fusion_visual_dir": "./outputs/fusion/visuals",
  "fusion_loss_log_dir": "./outputs/fusion/logs"
}
```

For the two-stage training protocol described in the paper, freeze the pretrained material branch during fusion training:

```json
{
  "freeze_material_in_fusion": true
}
```

Run:

```bash
python train_fusion.py
```

The fusion-stage checkpoints are saved to `fusion_save_dir`. The final checkpoint is saved as:

```text
fusion_final.pth
```

To resume fusion training, set:

```json
{
  "resume_fusion_weight_path": "./outputs/fusion/checkpoints/fusion_epoch_020.pth"
}
```

After training, copy or rename the selected checkpoint for testing:

```text
weights/fusion_weights.pth
```

## Testing

Prepare the test dataset:

```text
datasets/test/
├── ir/
│   ├── 0001.png
│   └── ...
└── vi/
    ├── 0001.png
    └── ...
```

Edit `params/default/fusion_test.json`:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "test_ir_dir": "./datasets/test/ir",
  "test_vis_dir": "./datasets/test/vi",
  "test_list": "",
  "fusion_weight_path": "./weights/fusion_weights.pth",

  "test_fused_save_dir": "./validation/marc_fusion/fused",
  "test_visual_save_dir": "./validation/marc_fusion/visual",

  "test_patch_size": 128,
  "test_stride": 64,
  "test_tile_border": 32,
  "test_use_hanning": true,

  "test_save_feature_maps": true,
  "test_save_feature_npy": false,
  "test_material_intervention": "none"
}
```

Run:

```bash
python test_fusion.py
```

The fused images are saved to:

```text
validation/marc_fusion/fused/
```

Side-by-side visualizations are saved to:

```text
validation/marc_fusion/visual/
```

When `test_save_feature_maps` is enabled, the following response maps are also saved:

```text
validation/marc_fusion/
├── zr/
├── ze/
├── c/
├── qir/
└── qvis/
```

where:

- `zr` is the reflection-associated material response.
- `ze` is the thermal-emission-associated material response.
- `c` is the low-frequency contrast/consistency response.
- `qir` is the infrared high-frequency reliability response.
- `qvis` is the visible high-frequency reliability response.

The test script uses overlapping tiled inference. The following condition must be satisfied:

```text
test_stride <= test_patch_size - 2 × test_tile_border
```

For example, the default setting `128 / 64 / 32` is valid.

To save only fused images and reduce storage use, set:

```json
{
  "test_save_feature_maps": false
}
```

## Output Summary

```text
outputs/
├── material/
│   ├── checkpoints/
│   ├── visuals/
│   └── logs/
└── fusion/
    ├── checkpoints/
    ├── visuals/
    └── logs/

validation/
└── marc_fusion/
    ├── fused/
    ├── visual/
    ├── zr/
    ├── ze/
    ├── c/
    ├── qir/
    └── qvis/
```

## Notes

- Keep the network configuration consistent across material training, fusion training, and testing.
- Avoid absolute local paths in released JSON files.
- Do not commit datasets, checkpoints, or generated outputs directly to the source-code repository.
- Large pretrained weights should be distributed through GitHub Releases, OSF, Zenodo, or another model-hosting service.
- If CUDA out-of-memory errors occur, reduce `batch_size` or `train_patch_size`.
- If mixed-precision execution causes a dtype error in the wavelet operations, set `"use_amp": false`.

## Citation

Citation information will be added after publication.
