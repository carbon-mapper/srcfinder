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


## FCN Pipeline

```
$ python fcn_pred_pipeline.py -h
usage: fcn_pred_pipeline.py [-h] [--pad PAD] [--band BAND] [--scale SCALE]
                            [--model MODEL] [--gpus GPUS [GPUS ...]]
                            [--batch BATCH] [--output OUTPUT]
                            flightline

Generate a flightline saliency map with a FCN.

positional arguments:
  flightline            Filepaths to flightline ENVI IMG.

optional arguments:
  -h, --help            show this help message and exit
  --pad PAD, -p PAD     Pad input by 0 or more pixels to avoid edge effects.
  --band BAND, -n BAND  Band to read if multiband
  --scale SCALE, -s SCALE
                        Downscaling factor of the model
  --model MODEL, -m MODEL
                        Model to use for prediction.
  --gpus GPUS [GPUS ...], -g GPUS [GPUS ...]
                        GPU devices for inference. -1 for CPU.
  --batch BATCH, -b BATCH
                        Batch size per device.
  --output OUTPUT, -o OUTPUT
                        Output directory for generated saliency maps.
```

### Summary

This pipeline generates a saliency map for each flightline by converting the CNN
into an FCN and using shift-and-stitch as described in
[(Long et al. 2015)](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf).

The saliency map is saved as an ENVI IMG in the directory specified by
`--output`. Its default value is `./`, the current working directory. The
saliency map has float values ranging from `[0,1]`, with `1` indicating the
presence of a methane plume.

This method involves loading the entire flightline into the VRAM of the GPU. For long flightlines, this may not be possible, event at batch size of 1. Either crop the flightline to be smaller or use a GPU with more VRAM. To help determine requirements, the script `fcn_mem_test.py` has been provided.

### Dependencies

The following packages must be installed via `pip`, along with the appropriate
GPU CUDA drivers.

* `numpy`
* `matplotlib`
* `tqdm`
* `rasterio`
* `pytorch`
* `torchvision`

### Examples

These examples were run on a 32-core `Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz`
server with `126 GB` of RAM and two Tesla M60 cards, effectively 4 GPUs with 
`2048` CUDA cores and `8GB` of VRAM each.

These examples include benchmarks for expected performance with different
hardware configurations. Notice that more GPUs and larger batch sizes do not
always result in the fastest runtime - there is some tradeoff due to memory
management.

An example product of running
```
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -g 0 1 2 3 -b 4 -n 4
```
is provided at `samples/ang20200924t211102_ch4mf_v2y1_img_CalCh4_v8+COVID_QC+Permian_QC_AA_saliency_pad256.img`

#### CPU Inference

```bash
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 8
[STEP] MODEL INITIALIZATION
[INFO] Finding model weightpath.
[INFO] Found /home/jakelee/2022/srcfinder/cnn/models/COVID_QC.pt.
[INFO] Initializing pytorch device.
[INFO] Loading model.
[INFO] Converting CNN to FCN.
[INFO] Initializing Dataloader.
[STEP] MODEL PREDICTION
...
// ETA 25 minutes
```

```bash
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 32 -n 4
// ETA 35 minutes
```

#### Single GPU Inference

```bash
// Single GPU inference with batch size 1
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 1 -n 4
// ETA 2.5 minutes
```

```bash
// Single GPU inference with batch size 4
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 4 -n 4
// ETA 2.5 minutes
```

```bash
// Single GPU inference with batch size 8
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 8 -n 4
// ETA 2.15 minutes
```

#### Quad GPU Inference

```bash
// Quad-GPU inference with batch size 1 each
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 1 -n 4
// ETA 2.5 minutes
```

```bash
// Quad-GPU inference with batch size 4 each
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 4 -n 4
// ETA 1.5 minutes
```

```bash
// Quad-GPU inference with batch size 8 each
$ python fcn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 8 -n 4
// ETA 1 minute
```

## CNN Pipeline

```
$ python cnn_pred_pipeline.py -h
usage: cnn_pred_pipeline.py [-h] [--model {COVID_QC,CalCH4_v8,Permian_QC}]
                            [--gpus GPUS [GPUS ...]] [--batch BATCH]
                            [--output OUTPUT]
                            flightline

Generate a flightline saliency map with a CNN.

positional arguments:
  flightline            Filepaths to flightline ENVI IMG.

optional arguments:
  -h, --help            show this help message and exit
  --model {COVID_QC,CalCH4_v8,Permian_QC}, -m {COVID_QC,CalCH4_v8,Permian_QC}
                        Model to use for prediction.
  --gpus GPUS [GPUS ...], -g GPUS [GPUS ...]
                        GPU devices for inference. -1 for CPU.
  --batch BATCH, -b BATCH
                        Batch size per device.
  --output OUTPUT, -o OUTPUT
                        Output directory for generated saliency maps.
```

### Summary

This pipeline generates a saliency map for each flightline by predicting a value
for each pixel with a CNN.

The saliency map is saved as an ENVI IMG in the directory specified by
`--output`. Its default value is `./`, the current working directory. The
saliency map has float values ranging from `[0,1]`, with `1` indicating the
presence of a methane plume.

### Dependencies

The following packages must be installed via `pip`, along with the appropriate
GPU CUDA drivers.

* `numpy`
* `matplotlib`
* `tqdm`
* `rasterio`
* `pytorch`
* `torchvision`

### Examples

These examples were run on a 32-core `Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz`
server with `126 GB` of RAM and two Tesla M60 cards, effectively 4 GPUs with 
`2048` CUDA cores and `8GB` of VRAM each.

These examples include benchmarks for expected performance with different
hardware configurations. Notice that more GPUs and larger batch sizes do not
always result in the fastest runtime - there is some tradeoff due to memory
management.

#### CPU Inference

```bash
// CPU inference with batch size 8
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 8 
[STEP] MODEL INITIALIZATION
[INFO] Finding model weightpath.
[INFO] Found /home/jakelee/2022/srcfinder/cnn/models/COVID_QC.pt.
[INFO] Initializing pytorch device.
[INFO] Loading model.
[INFO] Initializing Dataloader.
(1, 669, 2801)
[STEP] MODEL PREDICTION
Predicting shifts...
// ETA ~8 hours
```

```bash
// CPU inference with batch size 32
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 32
// ETA ~7 hours
```

#### Single GPU Inference

```bash
// Single GPU inference with batch size 8
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 8
// ETA ~2.3 hours
```

```bash
// Single GPU inference with batch size 32
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 32
// ETA ~1.3 hours
```

```bash
// Single GPU inference with batch size 512
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 512
// ETA ~1.3 hours
```

#### Quad GPU Inference

```bash
// Quad-GPU inference with batch size 8 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 8
// ETA ~7.5 hours
```

```bash
// Quad-GPU inference with batch size 32 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 32
// ETA ~2 hours
```

```bash
// Quad-GPU inference with batch size 512 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 512
// ETA ~0.5 hours
```

## Compute Performance Overview

This table summarizes the results in the examples above.

Additional benchmarks are provided with a different machine with the same CPU
and a P4 GPU, which is slightly faster than the M60. A comparison is included
below, including two GPU models available via AWS EC2.

Benchmarks on the two GPU models are not yet available, but may be added
in the future.

| Specs           | M60   | P4    | K80 (p2.xlarge) | V100 (P3.2xlarge) |
| --------------- | ----- | ----- | --------------- | ----------------- |
| CUDA cores      | 2048  | 2560  | 2492            | 5120              |
| FP32 TFLOPS     | 4.825 | 5.704 | 4.113           | 14.13             |
| VRAM (GB)       | 8     | 8     | 12              | 16                |
| mem bwth (GB/s) | 160   | 192   | 240             | 897               |


### FCN Pipeline Performance

**Note: Time in minutes, not hours.**

| CPU + GPU | GPUs | batch | Runtime (min) |
| --------  | ---- | ----- | ------------- |
| 32 + M60  | -1   | 8     | 25            |
| 32 + M60  | -1   | 32    | 35            |
| 32 + M60  | 0    | 1     | 2.5           |
| 32 + M60  | 0    | 4     | 2.5           |
| 32 + M60  | 0    | 8     | 2.2           |
| 32 + P4   | 0    | 8     | 2.0           |
| 32 + M60  | 0123 | 1     | 2.5           |
| 32 + M60  | 0123 | 4     | 1.5           |
| 32 + M60  | 0123 | 8     | 1.0           |

-1 indicates only using the CPUs, and 0+ indicates GPU devices.

### CNN Pipeline Performance

| CPU + GPU | GPUs | batch | Runtime (hrs) |
| --------- | ---- | ----- | ------------- |
| 32 + M60  | -1   | 8     | 8             |
| 32 + M60  | -1   | 32    | 7             |
| 32 + M60  | 0    | 8     | 2.3           |
| 32 + M60  | 0    | 32    | 1.3           |
| 32 + M60  | 0    | 512   | 1.3           |
| 32 + P4   | 0    | 512   | 1             |
| 32 + M60  | 0123 | 8     | 7.5           |
| 32 + M60  | 0123 | 32    | 2             |
| 32 + M60  | 0123 | 512   | 0.5           |

-1 indicates only using the CPUs, and 0+ indicates GPU devices.