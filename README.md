# MsD-UMamba

Official implementation workspace for **MsD-UMamba: Boundary-aware Multi-scale Mamba Network for Efficient Medical Image Segmentation**.



## Repository Layout

```text
.
├── .devcontainer/
│   ├── Dockerfile
│   └── devcontainer.json
├── data/MnMs2
│   ├── cache/
│   ├── raw/
│   └── MnM2.rar
├── results/mnms2
│   └── msd_umamba/
│       ├── model_epoch_<index>.pth
│       ├── train.pth
│       └── best.pth
├── datasets/
│   ├── _factory.py
│   ├── utils.py
│   └── heart/
│       ├── dataset.py
│       ├── mnms/mnms2.py
│       └── utils.py
├── models/
│   ├── layers/ss2d.py
│   ├── blocks/damamba.py
│   └── nets/MsD_UMamba.py
├── losses/
│   ├── __init__.py
│   └── segmentation.py
├── run/
│   ├── run.py
│   └── train.py
├── scripts/train.sh
├── requirements.txt
└── README.md
```

## Training

The intended development-runner invocation from the repository root is:

```bash
bash scripts/train.sh
```
