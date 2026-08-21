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
  "fusion_weight_path": "./weights/weights_marc_fusion_best.pth"
}
```

The released weight should be loaded as a PyTorch `state_dict`. For exact architecture matching, use strict loading:

```python
state_dict = torch.load(
    "./weights/weights_marc_fusion_best.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict, strict=True)
model.eval()
```

Please use the network configuration provided with the repository when evaluating the released checkpoint.
