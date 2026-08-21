# Pretrained Weights

The pretrained MARC-Fusion weights used for testing and reproduction are available from the official GitHub release:

https://github.com/YichenLiu156/MARC_Fusion/releases/tag/Weights

## Usage

1. Download the released MARC-Fusion weight file from the link above.
2. Place the weight file in this `weights/` directory.
3. If necessary, rename the released file to:

```text
fusion_weights.pth
```

The expected directory layout is:

```text
weights/
├── README.md
└── fusion_weights.pth
```

Then set the testing configuration to:

```json
{
  "fusion_weight_path": "./weights/marc_fusion_best.pth"
}
```


