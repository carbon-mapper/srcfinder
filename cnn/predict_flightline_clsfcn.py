import sys
import os
import os.path  as op
from pathlib    import Path
from datetime   import datetime
import argparse

import numpy    as np
from tqdm       import tqdm
import logging

import torch
from torch      import nn
from torchvision import transforms

import rasterio
from archs.googlenetAA import googlenetAA
import cmutils
import cmutils.pytorch as cmtorch
from cmutils.fcn import ShiftStitchDataset, stitch_stack, cnn_to_fcn

def build_dataloader(x, model, batch):
    """ Build a FCN dataloader based on provided arguments

    Parameters
    ----------
    fl_path: str
        Filepath to flightline
    band: int
        Band ID (1-indexed)
    pad: int
        Initial input zero-padding
    model: str
        Name of model
    batch: int
        Batch size, including multiplication with number of GPUs
    """

    transform = transforms.Compose([cmtorch.ClampScaleMethaneTile()])

    dataloader = torch.utils.data.DataLoader(
        ShiftStitchDataset(x, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=0
    )

    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flightline saliency map with a FCN."
    )

    parser.add_argument('flightline',       help="Filepaths to flightline ENVI IMG.",
                                            type=str)
    parser.add_argument('--pad', '-p',      help="Pad input by 0 or more pixels to avoid edge effects.",
                                            default=256,
                                            type=int)
    parser.add_argument('--band', '-n',     help="Band to read if multiband",
                                            default=4,
                                            type=int)
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                                            default="models/multicampaign_googlenet.pt")
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=1,
                                            type=int)
    parser.add_argument('--outroot', '-o',   help="Output directory for generated saliency maps.",
                                            default="pred_out",
                                            type=str)

    args = parser.parse_args()

    # SETUP ####################################################################

    # Set up output directories and files
    expname = f"fcnpred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"

    outdir = op.join(args.outroot, expname)
    cmutils.check_mkdir(args.outroot)
    cmutils.check_mkdir(op.join(outdir))

    # TODO: Given time, switch to tensorboard
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
    weightpath = '/home/jake/CMML/dl-pipelines/agu_weights/googlenet.pt'
    # weightpath = op.join(os.path.dirname(os.path.abspath(__file__)), 'models', 
    #                         args.model + ".pt")
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
    model = googlenetAA(pretrained=False, num_classes=2, init_weights=False)
    model.load_state_dict(torch.load(weightpath,map_location=device))
    fcn = cnn_to_fcn(model, truncate=-5, pool=(8,8), in_ch=1024)

    if len(args.gpus) > 1:
        # Multi-GPU
        fcn = fcn.to(device)
        fcn = nn.DataParallel(fcn, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        fcn = fcn.to(device)
    
    fcn.eval()


    # DATA #####################################################################

    logging.info("Reading data")
    dataset = rasterio.open(args.flightline)
    x = np.pad(dataset.read(args.band),args.pad,constant_values=-9999)

    logging.info("Initializing dataloader")
    dataloader = build_dataloader(
        x,
        args.model,
        args.batch * len(args.gpus)
    )


    # INFERENCE ################################################################

    logging.info("Running predictions")
    allpred = None
    ts = []
    ls = []
    for (t, l), batch in tqdm(dataloader, desc="FCN Pred"):
        inputs = batch.to(device)
        with torch.no_grad():
            preds = fcn(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = preds.cpu().detach().numpy()[:,1,:,:]
            if allpred is None:
                allpred = preds
            else:
                allpred = np.concatenate((allpred, preds))
            ts += t
            ls += l

    logging.info("Stitching shifts")
    allpred = stitch_stack(x.shape, ts, ls, allpred)
    allpred[x == -9999] = -9999
    if args.pad>0:
        allpred = allpred[args.pad:-args.pad,args.pad:-args.pad]

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

        outf = f"{Path(args.flightline).stem}_ch4saliency.img"
        logging.info(f"Saving to {op.join(outdir, outf)}")
        with rasterio.open(op.join(outdir, outf), 'w', **profile) as dst:
            dst.write(allpred.astype(rasterio.float32), 1)

    logging.info("Done!")
