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
pip install wandb
```
The only difference between the two environments is the `pytorch` channel `pytorch-cuda=12.1` vs. `cpuonly`.
Weights and Biases is used for experiment tracking, and is only needed for model training.

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
  - We have previously described this model as "UC-Net", as it trains on combined segmentation and classification loss.
- UPerNet: `archs`
  - UPerNet as described by [Xiao et al.](https://arxiv.org/abs/1807.10221)
  - Modified from implementation from [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py)
    - Replaced ConvNext with GoogLeNet backbone model
    - Replaced PSP pooling to use fixed kernel sizes instead of adaptive pooling
  - This model also trains on combined segmentation and classification loss.

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
```bash
python train_cls.py train_abspath_g3k.csv test_abspath_g3k.csv \
--lr 0.0001 \
--epochs 100 \
--batch 16 \
--outroot /path/to/outdir/ \
--gpu 0 \
--model googlenetAA \
--project 20231106_cls
```

To be used as:
- Model to be converted for FCN
- Model to be used as a backbone for UPerNet

#### `models/multicampaign_deepunet.pt`

Trained with
```bash
python train_unet_upnet.py labels/train_abspath_g3k.csv labels/test_abspath_g3k.csv \
--project 20240125_sweep \
--model DeepUNet \
--outroot outdir/ \
--gpu 0
```

#### `models/multicampaign_upernet.pt`

Trained with
```bash
python train_unet_upnet.py labels/train_abspath_g3k.csv labels/test_abspath_g3k.csv \
--project 20240125_sweep \
--model UPerNet \
--outroot outdir/ \
--gpu 0
```

## FCN Pipeline

### Training

This trains an antialiased GoogLeNet plume classification model for
conversion to FCN, or for use as a backbone for UPerNet.

```bash
% python train_cls.py -h
usage: train_cls.py [-h] [--project PROJECT] [--exp EXP] [--dataroot DATAROOT]
                    [--model {googlenetAA}] [--lr LR] [--epochs EPOCHS]
                    [--batch BATCH] [--outroot OUTROOT] [--no-sam] [--gpu GPU]
                    [--wandb-dir WANDB_DIR]
                    traincsv valcsv

Train a classification model on tiled methane data.

positional arguments:
  traincsv              Filepath of the training set CSV
  valcsv                Filepath of the validation set CSV

options:
  -h, --help            show this help message and exit
  --project PROJECT     Project name for wandb
  --exp EXP             Run name for wandb
  --dataroot DATAROOT   Root directory for relative paths. Defaults to / for
                        absolute paths.
  --model {googlenetAA}
                        Model architecture to train
  --lr LR               Learning rate
  --epochs EPOCHS       Epochs for training
  --batch BATCH         Batch size for model training
  --outroot OUTROOT     Root of output directories
  --no-sam              Disable SAM
  --gpu GPU             GPU index to use
  --wandb-dir WANDB_DIR
                        Output directory for wandb logs. Defaults to ./wandb
```

Notes:
- Model training accesses many small files very quickly.
  The dataset should be stored on-hardware if possible, not NFS.
- Model training should be done on a GPU machine for orders-of-magnitude
  speedup. The included pre-trained model was trained on G3K with 1 GPU
  (RTX 5000) in 4 hours.

### Flightline Inference

This converts a plume classification model into a FCN segementation model.
Given a CMF flightline, produces a saliency map ENVI IMG. Values from [0, 1]
indicate plume confidence.

_Backwards Compatibility: This is equivalent to `fcn_pred_pipeline.py` from the previous delivery._

```bash
% python predict_flightline_clsfcn.py -h
usage: predict_flightline_clsfcn.py [-h] [--pad PAD] [--band BAND]
                                    [--model MODEL] [--gpus GPUS [GPUS ...]]
                                    [--batch BATCH] [--outroot OUTROOT]
                                    flightline

Generate a flightline saliency map with a FCN.

positional arguments:
  flightline            Filepaths to flightline ENVI IMG.

options:
  -h, --help            show this help message and exit
  --pad PAD, -p PAD     Pad input by 0 or more pixels to avoid edge effects.
  --band BAND, -n BAND  Band to read if multiband
  --model MODEL, -m MODEL
                        Model to use for prediction.
  --gpus GPUS [GPUS ...], -g GPUS [GPUS ...]
                        GPU devices for inference. -1 for CPU.
  --batch BATCH, -b BATCH
                        Batch size per device.
  --outroot OUTROOT, -o OUTROOT
                        Output directory for generated saliency maps.
```

Notes:
- FCN is the most compute-intensive pipeline. Prior work showed that inference on a GPU
  took around 2 minutes, while inference on a CPU took about 30 minutes.
- FCN is the worst-performing pipeline compared to U-Net and UPerNet pipelines, as shown
  in [Lee et al.](https://doi.org/10.22541/essoar.170365353.38110853/v1)
- _Additional compute information to come_

## U-Net & UPerNet Pipeline

### Training

This trains a U-Net or UPerNet model for plume detection via segmentation.

```bash
% python train_unet_upnet.py -h
usage: train_unet_upnet.py [-h] [--project PROJECT] [--exp EXP]
                           [--dataroot DATAROOT] [--model {DeepUNet,UPerNet}]
                           [--lr LR] [--epochs EPOCHS] [--batch BATCH]
                           [--outroot OUTROOT] [--gpu GPU]
                           [--backbone BACKBONE] [--wandb-dir WANDB_DIR]
                           traincsv valcsv

Train a segmentation model on tiled methane data.

positional arguments:
  traincsv              Filepath of the training set CSV
  valcsv                Filepath of the validation set CSV

options:
  -h, --help            show this help message and exit
  --project PROJECT     Project name for wandb
  --exp EXP             Run name for wandb
  --dataroot DATAROOT   Root directory for relative paths. Defaults to / for
                        absolute paths.
  --model {DeepUNet,UPerNet}
                        Which model to train
  --lr LR               Learning rate, U-Net default 0.001, UPerNet default
                        0.0001
  --epochs EPOCHS       Epochs for training. Default 200.
  --batch BATCH         Batch size for model training
  --outroot OUTROOT     Root of output directories
  --gpu GPU             Specify GPU index to use
  --backbone BACKBONE   Filepath to backbone weights. Defaults to
                        models/multicampaign_googlenet.pt
  --wandb-dir WANDB_DIR
                        Output directory for wandb logs. Defaults to ./wandb
```

Notes:
- Training a DeepUNet model completely for 200 epochs may take up to 48 hours.
- A UPerNet model may only require 50 epochs to fully converge.
- Removing the per-epoch evaluation may improve training time, as it is non-trivial.
- UPerNet training relies on an existing classification model.

### Flightline Inference

This runs a U-Net or UPerNet directly on the entire cmf flightline.
Given a CMF flightline, produces a saliency map ENVI IMG. Values from [0, 1]
indicate plume confidence.

```bash
% python predict_flightline_unet_upnet.py -h
usage: predict_flightline_unet_upnet.py [-h] [--band BAND] [--weights WEIGHTS]
                                        [--arch {UperNet,DeepUNet}]
                                        [--gpus GPUS [GPUS ...]]
                                        [--outroot OUTROOT]
                                        flightline

Generate a flightline saliency map with a FCN.

positional arguments:
  flightline            Filepaths to flightline ENVI IMG.

options:
  -h, --help            show this help message and exit
  --band BAND, -n BAND  Band to read if multiband
  --weights WEIGHTS, -w WEIGHTS
                        Weights to use for prediction.
  --arch {UperNet,DeepUNet}, -a {UperNet,DeepUNet}
                        Arch to use for prediction.
  --gpus GPUS [GPUS ...], -g GPUS [GPUS ...]
                        GPU devices for inference. -1 for CPU.
  --outroot OUTROOT, -o OUTROOT
                        Output directory for generated saliency maps.
```

Notes:
- This pipeline can predict an entire flightline within seconds on a GPU.
- Even if the entire flightline cannot be loaded into a GPU VRAM, runtime on CPU is feasible.
- _Additional compute information to come_

## Archive

- `archive/posthoc_down.py`
  - Jake's methodology for downsampling airborne flightlines to spaceborne-like resolution.