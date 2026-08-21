# MARC-Fusion

This repository contains the PyTorch implementation accompanying the manuscript **"Material-aware selective fusion for infrared-visible images"**.

MARC-Fusion learns **material-related cross-modal evidence** from registered infrared-visible image pairs and uses it to support frequency-conditioned fusion decisions. The training procedure contains two stages:

1. **Material-related cross-modal evidence pretraining**
2. **Fusion-network training**

The released implementation also includes an overlapping tiled-inference utility for convenient testing on large images. For exact reproduction of the paper-reported evaluation protocol, follow the inference settings described in the manuscript and supplementary material.

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
- Material-stage training requires at least two paired samples because negative samples are drawn from a different registered image pair.

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

The released MARC-Fusion weight is available from the GitHub release:

> **[MARC-Fusion V1.0.0 pretrained weights](https://github.com/YichenLiu156/MARC_Fusion/releases/tag/Weights)**

Place the released PyTorch `state_dict` in the `weights/` directory and rename it when necessary:

```text
weights/
└── fusion_weights.pth
```

The test configuration should contain:

```json
{
  "fusion_weight_path": "./weights/marc_fusion_best.pth"
}
```

Load the released weight with:

```python
state_dict = torch.load(
    "./weights/marc_fusion_best.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict, strict=True)
model.eval()
```

The released file is intended to contain the model `state_dict` only; optimizer states, local dataset paths, and training logs are not required for inference.

## Configuration

The scripts use JSON configuration files. Before running a script, check the `json_path` variable near the bottom of that script and ensure that it points to the intended configuration file.

Recommended configuration files:

```text
params/default/train_material.json
params/default/train_fusion.json
params/default/fusion_test.json
params/default/fusion_network.json
```

The recommended organization separates responsibilities:

- `fusion_network.json`: network architecture and model-side forward settings.
- `train_material.json`: Stage-1 training settings.
- `train_fusion.json`: Stage-2 training settings.
- `fusion_test.json`: test data, checkpoint, and inference/output settings.

Avoid duplicating architecture fields in `fusion_test.json` unless an intentional experiment-specific override is required. When configuration merging is enabled, test- or train-level values may override fields loaded from `fusion_network.json`.

Paths inside JSON files are resolved relative to the directory from which the Python process is started. Using paths relative to the repository root is recommended.

## Training

The paper uses RoadScene for both training stages with random `128 x 128` crops.

### Stage 1: Material-Related Cross-Modal Evidence Pretraining

Edit `params/default/train_material.json`.

For the paper-reported setting, use:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "train_ir_dir": "./datasets/train/ir",
  "train_vis_dir": "./datasets/train/vi",
  "train_list": "",

  "material_epochs": 30,
  "batch_size": 8,
  "train_patch_size": 128,
  "neg_patch_size": 128,

  "material_save_dir": "./outputs/material/checkpoints",
  "material_visual_dir": "./outputs/material/visuals",
  "material_loss_log_dir": "./outputs/material/logs",
  "resume_material_weight_path": ""
}
```

The principal Stage-1 optimization settings reported in the supplementary material are:

```text
Optimizer        : Adam
Learning rate    : 5e-5
Weight decay     : 0
Gradient clipping: 1
Epochs           : 30
Batch size       : 8
Patch size       : 128 x 128
```

The material objective uses the reported weights:

```text
lambda_rec = 1.0
lambda_grad^m = 0.5
lambda_c = 0.1
tau = 0.2
```

Negative samples for the stability contrastive loss are drawn from a different registered image pair.

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
  "resume_material_weight_path": "./outputs/material/checkpoints/material_epoch_XXX.pth"
}
```

### Stage 2: Fusion-Network Training

After Stage-1 pretraining, edit `params/default/train_fusion.json` and point `pretrained_material_weight_path` to the Stage-1 checkpoint.

For the paper-reported setting, use:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "train_ir_dir": "./datasets/train/ir",
  "train_vis_dir": "./datasets/train/vi",

  "pretrained_material_weight_path": "./outputs/material/checkpoints/material_final.pth",
  "resume_fusion_weight_path": "",

  "fusion_epochs": 133,
  "batch_size": 4,
  "train_patch_size": 128,
  "freeze_material_in_fusion": true,

  "fusion_save_dir": "./outputs/fusion/checkpoints",
  "fusion_visual_dir": "./outputs/fusion/visuals",
  "fusion_loss_log_dir": "./outputs/fusion/logs"
}
```

The principal Stage-2 optimization settings reported in the supplementary material are:

```text
Optimizer        : Adam
Learning rate    : 2e-5
Weight decay     : 0
Gradient clipping: 1
Batch size       : 4
Patch size       : 128 x 128
```

The material-related cross-modal evidence-acquisition branch is frozen during Stage-2 fusion training.

The checkpoint used for the results reported in the paper is the **fusion checkpoint at epoch 133**. If training is resumed from an intermediate checkpoint, preserve the effective epoch numbering so that the selected reported checkpoint remains identifiable.

Run:

```bash
python train_fusion.py
```

The fusion-stage checkpoints are saved to `fusion_save_dir`. A final checkpoint may also be saved as:

```text
fusion_final.pth
```

To resume fusion training, set for example:

```json
{
  "resume_fusion_weight_path": "./outputs/fusion/checkpoints/fusion_epoch_020.pth"
}
```

For paper-result reproduction, use the checkpoint corresponding to epoch 133 or the released V1.0.0 checkpoint supplied above.

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

A minimal test configuration is:

```json
{
  "network_config_path": "./params/default/fusion_network.json",
  "test_ir_dir": "./datasets/test/ir",
  "test_vis_dir": "./datasets/test/vi",
  "test_list": "",
  "fusion_weight_path": "./weights/fusion_weights.pth",

  "test_fused_save_dir": "./validation/marc_fusion/fused",
  "test_visual_save_dir": "./validation/marc_fusion/visual",

  "test_save_feature_maps": false,
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

### Paper-reported inference protocol

For the evaluation reported in the manuscript and supplementary material, images are evaluated at their **original spatial resolution**, with only the padding required by the wavelet decomposition/reconstruction. No post-processing is applied.

The paper-reported protocol should therefore be treated as the reference setting for metric reproduction.

### Optional tiled inference utility

The released test code also provides overlapping tiled inference for large images. This utility is convenient when full-resolution inference is constrained by GPU memory, but it should not be confused with the paper-reported original-resolution evaluation protocol.

When tiled inference is enabled, the current implementation uses parameters such as:

```json
{
  "test_patch_size": 128,
  "test_stride": 64,
  "test_tile_border": 32,
  "test_use_hanning": true
}
```

The following condition must be satisfied:

```text
test_stride <= test_patch_size - 2 x test_tile_border
```

For example, `128 / 64 / 32` satisfies this condition.

Because overlapping tiles, border cropping, and Hann-window blending can change the numerical output relative to direct full-resolution inference, do not use tiled output as a substitute for the paper-reported protocol when reproducing the published metrics unless explicitly documented for a separate experiment.

### Optional response-map export

When `test_save_feature_maps` is enabled, the following response maps can be saved:

```text
validation/marc_fusion/
├── zr/
├── ze/
├── c/
├── qir/
└── qvis/
```

where:

- `zr` is the reflection-response-biased material-related response.
- `ze` is the thermal-response-biased material-related response.
- `c` is the low-frequency cross-modal response contrast map.
- `qir` is the infrared high-frequency reliability response.
- `qvis` is the visible high-frequency reliability response.

To save only fused images and reduce storage use, set:

```json
{
  "test_save_feature_maps": false
}
```

## Reproducibility Notes

For consistency with the paper, the principal settings are summarized below:

```text
Stage 1 dataset           : RoadScene
Stage 1 patch size        : 128 x 128
Stage 1 batch size        : 8
Stage 1 epochs            : 30
Stage 1 learning rate     : 5e-5

Stage 2 dataset           : RoadScene
Stage 2 patch size        : 128 x 128
Stage 2 batch size        : 4
Stage 2 learning rate     : 2e-5
Material branch in Stage 2: frozen
Reported fusion checkpoint: epoch 133

Weight decay              : 0
Gradient clipping         : 1
Inference                 : original resolution
Padding                   : wavelet-required padding only
Post-processing           : none
```

For exact checkpoint portability, keep the network architecture configuration synchronized with the configuration used to produce the released checkpoint. In particular, forward-affecting parameters should not be silently overridden by an experiment-specific test JSON.

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

## Citation

Citation information will be added after publication.
