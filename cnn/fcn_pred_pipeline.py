import sys
import os
import os.path as op
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import rasterio
from archs.googlenet1 import googlenet as googlenet_old
from archs.googlenetAA import googlenet

class ClampCH4(object):
    """ Preprocessing step for the methane layer """
    def __init__(self, vmin=250, vmax=4000):
        assert isinstance(vmin,int) and isinstance(vmax,int) and vmax > vmin
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, T):
        return torch.clamp(T, self.vmin, self.vmax)

    def __repr__(self):
        return self.__class__.__name__ + '(vmin={0}, vmax={1})'.format(self.vmin, self.vmax)

class FlightlineShiftStitch(torch.utils.data.Dataset):
    """ Single flightline for shift and stitching
    
    Usage:
    FlightlineShiftStitch(flightlinepath, transform, scale)
    """
    
    def __init__(self, x, transform, scale=32, band=1):
        self.x = x
        self.x_shape = x.shape
        self.transform = transform
        self.scale = scale

        pad0 = scale - (self.x_shape[0] % self.scale)
        pad1 = scale - (self.x_shape[1] % self.scale)

        # Left Right Top Bottom
        self.div_pad = nn.ZeroPad2d((0, pad1, 0, pad0))

    def __len__(self):
        return self.scale ** 2
    
    def __getitem__(self, idx):
        # Calculate shift-and-stitch padding for this index
        top = idx // self.scale
        left = idx % self.scale
        
        t = torch.as_tensor(self.x, dtype=torch.float).unsqueeze(0)
        if self.transform is not None:
            t = self.transform(t)

        # Divisibility padding
        t = self.div_pad(t)

        # Shift-and-Stitch padding
        # Left Right Top Bottom
        t = nn.ZeroPad2d((left, self.scale-left, top, self.scale-top))(t)
        return (top, left), t 

def stitch_stack(fl_shape, ts, ls, predstack, scale=32):
    """ Interlace shifted outputs

    fl_shape: Shape of original flightline for cropping
    ts: List of top shifts
    ls: List of left shifts
    predstacK: Stack of shifted predictions
    scale: Downscale factor of model, default 32.
    """
    # Storage for final stitched output
    stitched = np.zeros(shape=(predstack.shape[1]*scale, predstack.shape[2]*scale))
    
    # Iterate through shifts and outputs
    for i in range(predstack.shape[0]):
        top = ts[i]
        left = ls[i]
        # Save them to corresponding strided pixels
        stitched[scale-top-1::scale, scale-left-1::scale] = predstack[i]

    # Crop the top left
    #stitched = stitched[:fl_shape[0], :fl_shape[1]]
    # Crop the center
    #stitched = stitched[scale//2:fl_shape[0]+scale//2, scale//2:fl_shape[1]+scale//2]
    # Crop the bottom right
    stitched = stitched[scale:fl_shape[0]+scale, scale:fl_shape[1]+scale]

    return stitched


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
                                            default=1,
                                            type=int)
    parser.add_argument('--scale', '-s',    help="Downscaling factor of the model",
                                            default=32,
                                            type=int)
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                                            default="CalCh4_v8+COVID_QC+Permian_QC_AA")
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=8,
                                            type=int)
    parser.add_argument('--output', '-o',   help="Output directory for generated saliency maps.",
                                            default=".",
                                            type=str)

    args = parser.parse_args()


    # Initial model setup/loading
    print("[STEP] MODEL INITIALIZATION")

    print("[INFO] Finding model weightpath.")
    weightpath = op.join(os.path.dirname(os.path.abspath(__file__)), 'models', 
                            args.model + ".pt")
    if op.isfile(weightpath):
        print(f"[INFO] Found {weightpath}.")
    else:
        print(f"[INFO] Model not found at {weightpath}, exiting.")
        sys.exit(1)

    print("[INFO] Initializing pytorch device.")
    if args.gpus == [-1]:
        # CPU
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # apple m1/m2
            device = torch.device("mps")
        else:
            if not torch.cuda.is_available():
                print("[ERR] CUDA not found, exiting.")
                sys.exit(1)
        
            # Set first device
            device = torch.device(f"cuda:{args.gpus[0]}")
    print(f'[INFO] using device: {device}')
    
    print("[INFO] Loading model.")
    if "AA" not in args.model:
        model = googlenet_old(pretrained=False, num_classes=2, init_weights=False)
    else:
        model = googlenet(pretrained=False, num_classes=2, init_weights=False)
    model.load_state_dict(torch.load(weightpath,map_location=device))
    
    # FCN model setup
    print("[INFO] Converting CNN to FCN.")
    fcn = nn.Sequential(*list(model.children())[:-5]).to(device)
    # Size depends on training tile size
    # 256 = (8,8)
    # 64 = (2,2)
    fcn.add_module('pool_repl', nn.AvgPool2d((8,8), 1, padding=4, ceil_mode=False, count_include_pad=False))
    fcn.add_module('pool_crop', nn.ConstantPad2d((0, -1, 0, -1), 0))
    fcn.add_module('final_conv', nn.Conv2d(1024, 2, kernel_size=1).to(device))
    fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    if len(args.gpus) > 1:
        # Multi-GPU
        fcn = fcn.to(device)
        fcn = nn.DataParallel(fcn, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        fcn = fcn.to(device)
    
    fcn.eval()

    print("[INFO] Initializing Dataloader.")
    # Transform and dataloader
    if args.model == "COVID_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[110.6390],
                std=[183.9152]
            )]
        )
    elif args.model == "CalCH4_v8":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[140.6399],
                std=[237.5434]
            )]
        )
    elif args.model == "Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[100.2635],
                std=[158.7060]
            )]
        )
    else:
        print(f'[INFO] Using multicampaign norm factors')
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[115.0],
                std=[190.0]
            )]
        )

    dataset = rasterio.open(args.flightline)
    pad = args.pad
    x = np.pad(dataset.read(args.band),pad,constant_values=-9999)

    dataloader = torch.utils.data.DataLoader(
        FlightlineShiftStitch(
            x,
            transform=transform,
            scale=args.scale,
        ),
        batch_size=args.batch * len(args.gpus),
        shuffle=False,
        num_workers=0
    )

    cmfbase = op.split(args.flightline)[1]
    print("[STEP] MODEL PREDICTION")

    # Run shift predictions
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

    # Stitch
    print("[INFO] Stitching shifts.")
    allpred = stitch_stack(x.shape, ts, ls, allpred, scale=args.scale)
    allpred[x == -9999] = -9999
    if pad>0:
        allpred = allpred[pad:-pad,pad:-pad]

    # Save
    print("[STEP] RESULT EXPORT")
    with rasterio.Env():
        profile = dataset.profile

        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )

        outf = f"{Path(args.flightline).stem}_{args.model}_saliency_pad{pad}.img"
        print(f"[INFO] Saving to", op.join(args.output, outf))
        with rasterio.open(op.join(args.output, outf), 'w', **profile) as dst:
            dst.write(allpred.astype(rasterio.float32), 1)

    print("Done!")
