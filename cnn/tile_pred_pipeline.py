import sys
import os, csv
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
from archs.googlenet1 import googlenet
from sklearn.metrics import classification_report

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

class TiledDatasetClass1Ch(torch.utils.data.Dataset):
    """ Classification dataset only using the methane channel
    
    Usage:
    TiledDatasetClass1Ch([[path, label], [path, label], ...], ...)
    """

    def __init__(self, datacsv, transform):
        self.datarows = []
        dataroot = op.split(datacsv)[0]
        with open(datacsv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                # Correct absolute path to relative path
                imgfile = op.join(dataroot,row[0])
                # 1/0 label (-1 is 0)
                imglabel = 1 if int(row[1]) == 1 else 0
                self.datarows.append([imgfile,imglabel])
                
        self.transform = transform

    def __len__(self):
        return len(self.datarows)

    def __getitem__(self, idx):
        # Absolute path to image
        x_path, y = self.datarows[idx]
        x_img = rasterio.open(x_path)
        x = np.expand_dims(x_img.read(x_img.count), axis=0)

        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        return (x, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Compute tilewise CNN predictions."
    )

    parser.add_argument('tilecsv',       help="CSV file containing relative paths to labeled tiles",
                                            type=str)
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                                            default="COVID_QC",
                                            choices=["COVID_QC", "CalCH4_v8", "Permian_QC", "CalCh4_v8+COVID_QC+Permian_QC"])
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=32,
                                            type=int)
    parser.add_argument('--output', '-o',   help="Output directory for generated saliency maps.",
                                            default=".",
                                            type=str)

    args = parser.parse_args()


    csvbase = op.splitext(op.split(args.tilecsv)[1])[0]
    predcsvf = '_'.join([csvbase,args.model,'predictions.csv'])
    reportf = '_'.join([csvbase,args.model,'report.txt'])
    

    # Initial model setup/loading
    print("[STEP] MODEL INITIALIZATION")

    print("[INFO] Finding model weightpath.")
    weightpath = op.join(Path(__file__).parent.resolve(), 'models', f"{args.model}.pt")
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
    model = googlenet(pretrained=False, num_classes=2, init_weights=False)
    model.load_state_dict(torch.load(weightpath,map_location=device))
    
    if len(args.gpus) > 1:
        # Multi-GPU
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        model = model.to(device)
    
    model.eval()

    print("[INFO] Initializing Dataloader.")

    crop = 256
    # Transform and dataloader
    if args.model == "COVID_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
	    transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=[110.6390],
                std=[183.9152]
            )]
        )
    elif args.model == "CalCH4_v8":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
	    transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=[140.6399],
                std=[237.5434]
            )]
        )
    elif args.model == "Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
	    transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=[100.2635],
                std=[158.7060]
            )]
        )
    elif args.model == "CalCh4_v8+COVID_QC+Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
	    transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=[115.0],
                std=[190.0]
            )]
        )

    dataloader = torch.utils.data.DataLoader(
        TiledDatasetClass1Ch(
            args.tilecsv,
            transform=transform,
        ),
        batch_size=args.batch * len(args.gpus),
        shuffle=False,
        num_workers=4 * len(args.gpus)
    )

    print("[STEP] MODEL PREDICTION")

    # Collect model predictions
    allpred = []
    datafiles,datalabs = np.array(dataloader.dataset.datarows).T
    for inputs,targets in tqdm(dataloader, desc="CNN Pred"):
        inputs = inputs.to(device)
        with torch.no_grad():
            preds = model(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            allpred += [x[1] for x in preds.cpu().detach().numpy()]

    allpred = np.array(allpred)
    datalabs = np.array(datalabs,dtype=int)
    predlabs = np.array(allpred>0.5,dtype=int)
    
    # Save
    print(f"[INFO] Saving to",predcsvf)
    np.savetxt(op.join(args.output,predcsvf),
               np.c_[datafiles,datalabs,allpred],delimiter=',',
               fmt='%s',header='path,label,pred')

    with open(op.join(args.output,reportf), 'w') as f:
        f.write(classification_report(datalabs, predlabs))
    print("Done!")
