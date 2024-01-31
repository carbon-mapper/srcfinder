# Carbon Mapper CNN

This directory contains pipelines for deep learning detection of methane plumes.

Last Updated 2024-01-30

## Environment and Dependencies

### Conda/Mamba

Two `conda`/`mamba` environments are provided depending on system resources. Mamba is strongly recommended
due to the complexity of the solve.

```bash
mamba env create -f environment_gpu_lite.yml  # For GPU-available systems
mamba env create -f encironment_cpu_lite.yml  # For CPU-available systems
```
The only difference between the two environments is the `pytorch` channel `pytorch-cuda=12.1` vs. `cpuonly`.

_Note: the previous [environment.yml](https://github.com/carbon-mapper/srcfinder/blob/258ce398a445ead4e295ab9db3dce7c080eedcff/cnn/environment.yml) is likely compatible._

### cmutils

CMutils is an installable package for common utility functions for Carbon Mapper CNN.

#### Installation

```bash
cd cmutils
pip install -e .  # Editable installation
```

#### Usage

Functions can be imported:
```python
import cmutils                      # General utility functions
import cmutils.pytorch as cmtorch   # Pytorch functions
```

There are also CLI scripts included:
```bash
$ plot_cmf -h
usage: plot_cmf.py [-h] [--minppmm MINPPMM] [--maxppmm MAXPPMM]
                   [--maxrdn MAXRDN] [--labpath LABPATH] [--rgbpath RGBPATH]
                   [--outdir OUTDIR] [--no-gui]
                   cmfpath

positional arguments:
  cmfpath            CMF image path

options:
  -h, --help         show this help message and exit
  --minppmm MINPPMM  Minimum CH4 enhancement (ppmm). Default 0
  --maxppmm MAXPPMM  Maximum CH4 enhancement (ppmm). Default 1500
  --maxrdn MAXRDN    Maximum radiance for scaling rgb channels. Default 15
  --labpath LABPATH  Override CMF label image path. Default looks for *.png or
                     *_mask.png
  --rgbpath RGBPATH  Override CMF rgb image path. Defaults to first 3 channels
                     of CMF if available, otherwise zeros.
  --outdir OUTDIR    Output directory for quicklook PDF
  --no-gui           Disable GUI show for server CLI use
```

```bash
$ duptiles_by_ptype -h
usage: duptiles_by_ptype.py [-h] [--flext FLEXT] [--cores CORES]
                            tileset flset ptype outdir

Duplicate tiles with new flightlines

positional arguments:
  tileset               Path to original tiled dataset
  flset                 Path to where new flightline products are located
  ptype                 Product type (e.g., _ch4mf_).
  outdir                Output directory. Must not exist.

options:
  -h, --help            show this help message and exit
  --flext FLEXT         Expected file extension of new flightline files. None
                        by default.
  --cores CORES, --c CORES
                        Number of parallel jobs
```

## Resources

### Architectures

- GoogLeNetAA: `archs/googlenetAA.py`
  - Original Inception image classification architecture as described by [Szegedy et al.](https://arxiv.org/abs/1409.4842)
  - Modified from original [pytorch implementation](https://pytorch.org/hub/pytorch_vision_googlenet/)
  - Included antialiasing for improved shift-invariance as described by [Zhang et al.](https://richzhang.github.io/antialiased-cnns/)
- FCN: `cmutils.fcn.cnn_to_fcn()`
  - Conversion of classification CNN to segmentation FCN as described by [Long et al.](https://arxiv.org/abs/1411.4038)
  - Other utility functions included in `cmutils.fcn` for data preparation and stitching
- U-Net: `archs/unet.py`
  - Convolutional network(s) for segmentation as described by [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
  - Variant delivered is `unet.DeepPaddedUNet`, which adds an additional double convolution block.
- UPerNet: `archs`
  - UPerNet as described by [Xiao et al.](https://arxiv.org/abs/1807.10221)
  - Modified from implementation from [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py)
    - Replaced ConvNext with GoogLeNet backbone model
    - Replaced PSP pooling to use fixed kernel sizes instead of adaptive pooling

### Labels

Labeled datasets are defined as absolute paths to datasets stored on G3K

- COVID_QC_no_overlap_[train/test]_abspath_g3k.csv
- CalCH4_v8_no_overlap_[train/test]_abspath_g3k.csv
- Permian_QC_[train/test]_abspath_g3k.csv
- gao_carb_2020_[train/test]_abspath_g3k.csv
- [train/test]_abspath_g3k_ca_only.csv
  - COVID_QC, CalCH4_v8, and gao_carb_2020 combined
- [train/test]_abspath_g3k.csv
  - All four campaigns combined

### Pre-trained Models

#### `models/multicampaign_googlenet.pt`

Trained with
```
python train_classifier.py train_abspath_g3k.csv test_abspath_g3k.csv \
--lr 0.0001 \
--epochs 100 \
--batch 16 \
--outroot /path/to/outdir/
--gpu 0
--model googlenetAA
--project 20231106_cls
```

To be used as:
- Model to be converted for FCN
- Model to be used as a backbone for UPerNet

#### `models/multicampaign_DeepUNet.pt`

_to be added_

#### `models/multicampaign_UPerNet.pt`

_to be added_