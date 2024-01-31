""" Flightline inference script

This script was designed to run on supercomputers/clusters.

v1: 2023-? jakelee - Converted from training script for inference
v2: 2024-01-30 jakelee - Cleanup for delivery

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""

import sys
import os.path as op
import argparse
import logging
from pathlib import Path

import numpy as np

import torch
from torchvision    import transforms

import rasterio
from archs.unet import DeepPaddedUNet
from archs.upernet import UperNet
import cmutils.pytorch as cmtorch

from sam.example.utility.step_lr import StepLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flightline saliency map with a FCN."
    )

    parser.add_argument('flightline',       help="Filepaths to flightline ENVI IMG.",
                                            type=str)
    parser.add_argument('--band', '-n',     help="Band to read if multiband",
                                            default=4,
                                            type=int)
    parser.add_argument('--weights', '-w',  help="Weights to use for prediction.")
    parser.add_argument('--arch', '-a',     help="Arch to use for prediction.",
                                            choices=["UperNet", "DeepUNet"])
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--outroot', '-o',   help="Output directory for generated saliency maps.",
                                            default="pred_out",
                                            type=str)

    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    # SETUP ####################################################################

    outdir = args.outroot

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(op.join(outdir, 'out.log')),
            logging.StreamHandler()
        ]
    )

    # MODEL ####################################################################

    logging.info("Model Initialization...")
    weightpath = args.weights
    if op.isfile(weightpath):
        logging.info(f"Found {weightpath}.")
    else:
        logging.info(f"Model not found at {weightpath}, exiting.")
        sys.exit(1)


    logging.info("Initializing pytorch device.")
    if args.gpus == [-1]:
        # CPU
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # apple m1/m2
            device = torch.device("mps")
        else:
            if not torch.cuda.is_available():
                logging.error("CUDA not found, exiting.")
                sys.exit(1)
            # Set first device
            device = torch.device(f"cuda:{args.gpus[0]}")
    logging.info(f"Using device: {device}")


    logging.info("Loading model")
    if args.arch == "DeepUNet":
        model = DeepPaddedUNet(in_ch=1)
    elif args.arch == "UperNet":
        model = UperNet(
            num_classes=1,
            in_channels=1,
            freeze_backbone=True
        )
    model.load_state_dict(torch.load(weightpath,map_location=device))
    model = model.to(device)
    model.eval()


    # DATA #####################################################################

    # Transform and dataloader
    transform = transforms.Compose([
        cmtorch.ClampScaleMethaneTile()
    ])

    logging.info("Reading data")
    dataset = rasterio.open(args.flightline)
    flightline = dataset.read(args.band)
    
    # Divisibility padding for each network
    if args.arch == "DeepUNet":
        div_ax0 = 32 - flightline.shape[0] % 32
        div_ax1 = 32 - flightline.shape[1] % 32
    else:
        div_ax0 = 16 - flightline.shape[0] % 16
        div_ax1 = 16 - flightline.shape[1] % 16

    pad_ax0 = (div_ax0 // 2, (div_ax0 // 2) + (div_ax0 % 2))
    pad_ax1 = (div_ax1 // 2, (div_ax1 // 2) + (div_ax1 % 2))

    x = np.pad(flightline, (pad_ax0, pad_ax1), constant_values=-9999)
    x = np.expand_dims(x, axis=[0,1])
    x = torch.tensor(x, dtype=torch.float).to(device)
    x = transform(x)

    # INFERENCE ################################################################

    logging.info("Running predictions")

    with torch.no_grad():
        pred = model(x)
        pred = torch.nn.functional.sigmoid(pred)
        pred = pred.cpu().detach().numpy()


    pred = pred[0,0,pad_ax0[0]:-pad_ax0[1],pad_ax1[0]:-pad_ax1[1]]
    pred[flightline == -9999] = -9999

    logging.info("Writing products")
    with rasterio.Env():
        profile = dataset.profile

        if 'blockysize' in profile:
            del profile['blockysize']
        if 'interleave' in profile:
            del profile['interleave']

        profile.update(
            dtype=rasterio.float32,
            count=1
        )

        outf = f"{Path(args.flightline).stem}_{args.arch}_saliency.img"
        logging.info(f"Saving to {op.join(outdir, outf)}")
        with rasterio.open(op.join(outdir, outf), 'w', **profile) as dst:
            dst.write(pred.astype(rasterio.float32), 1)

    logging.info("Done!")