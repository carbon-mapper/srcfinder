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

from archs.googlenetAA import googlenet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flightline saliency map with a FCN."
    )

    parser.add_argument('width',            help="Width of the simulated flightline input.",
                                            type=int)
    parser.add_argument('length',           help="Length of the simulated flightline input.",
                                            type=int)

    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=1,
                                            type=int)

    args = parser.parse_args()


    # Initial model setup/loading
    print("[STEP] MODEL INITIALIZATION")

    print("[INFO] Finding model weightpath.")

    weightpath = "models/multi_256AA.pt"
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
        if not torch.cuda.is_available():
            print("[ERR] CUDA not found, exiting.")
            sys.exit(1)
        
        # Set first device
        device = torch.device(f"cuda:{args.gpus[0]}")

    print("[INFO] Loading model.")
    model = googlenet(pretrained=False, num_classes=2, init_weights=False, num_channels=1)
    model.load_state_dict(torch.load(weightpath))
    
    # FCN model setup
    print("[INFO] Converting CNN to FCN.")
    fcn = nn.Sequential(*list(model.children())[:-5]).to(device)
    # Size depends on training tile size
    # 256 = (8,8)
    # 64 = (2,2)
    fcn.add_module('pool_repl', nn.AvgPool2d((8,8), 1, padding=4, ceil_mode=False, count_include_pad=False))
    #fcn.add_module('pool_repl', nn.AvgPool2d((2,2), 1, padding=1, ceil_mode=False, count_include_pad=False))
    fcn.add_module('pool_crop', nn.ConstantPad2d((0, -1, 0, -1), 0))
    fcn.add_module('final_conv', nn.Conv2d(1024, 2, kernel_size=1).to(device))
    fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    # fcn = nn.Sequential(*list(model.children())[:-2]).to(device)
    # fcn.add_module('final_conv', nn.Conv2d(256, 2, kernel_size=1).to(device))
    # fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    # fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    if len(args.gpus) > 1:
        # Multi-GPU
        fcn = fcn.to(device)
        fcn = nn.DataParallel(fcn, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        fcn = fcn.to(device)
    
    fcn.eval()

    print(f"[INFO] Initializing input size ({args.batch}, 1, {args.length}, {args.width}).")
    load = torch.ones(args.batch, 1, args.length, args.width).to(device)
    load = nn.ZeroPad2d((0, 32, 0, 32))(load)
    
    with torch.no_grad():
        out = fcn(load)
    
    print("SUCCESS")