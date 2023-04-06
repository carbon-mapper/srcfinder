import sys
import os
import os.path as op
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import rasterio
from archs.googlenet import googlenet

from scipy.interpolate import griddata
#sys.path.append(os.getenv('SRCFINDER_ROOT'))
#from srcfinder_util import *

class ClampCH4(object):
    """ Preprocessing step for the methane layer """
    def __init__(self, vmin=0, vmax=4000):
        assert isinstance(vmin,int) and isinstance(vmax,int) and vmax > vmin
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, T):
        return torch.clamp(T, self.vmin, self.vmax)

    def __repr__(self):
        return self.__class__.__name__ + '(vmin={0}, vmax={1})'.format(self.vmin, self.vmax)

def extrema(a,**kwargs):
    return np.nanmin(a),np.nanmax(a)

class ImageTiler(torch.utils.data.Dataset):
    def __init__(self, imgdata, transform, step=1, dim=256):
        """
        
        """
        self.transform = transform
        self.dim = dim
        self.hdim = dim//2
        self.step = step
        
        self.nrows = imgdata.shape[1]
        self.ncols = imgdata.shape[2]
        self.srows = self.nrows//self.step
        self.scols = self.ncols//self.step

        self.x = torch.as_tensor(imgdata, dtype=torch.float)
        if self.transform is not None:
            self.x = self.transform(self.x)
        print(f'imgdata.shape: {imgdata.shape}')
        print(f'total pixels: {self.nrows*self.ncols}')
        print(f'step pixels:  {self.srows*self.scols}')
        # pad to center fcn on tile centers
        self.x = transforms.Pad([self.hdim,self.hdim, self.hdim+1, self.hdim+1],
                                fill=0, padding_mode='constant')(self.x)
        print(f'pad(imgdata,[hdim,hdim,hdim+1,hdim+1]).shape: {self.x.size()}')
        
    def __len__(self):
        return self.srows*self.scols
    
    def __getitem__(self, idx):
        # x padded by dim//2 -> tij = dim x dim tile centered on pixel (i,j)  
        i = (idx // self.scols)*self.step
        j = (idx  % self.scols)*self.step
        tij = self.x[:,i:i+self.dim, j:j+self.dim]
        return tij,np.c_[[i],[j]]
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CNN to an FCN for flightline predictions.")
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                        default="COVID_QC",
                        choices=["COVID_QC", "CalCH4_v8",
                                 "Permian_QC", "CalCh4_v8+COVID_QC+Permian_QC"])
    parser.add_argument('--step', '-s',    help="Step size.",
                                            default=1,
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=32,
                                            type=int)
    parser.add_argument('--workers', '-w',    help="Number of parallel workers.",
                                            default=1,
                                            type=int)
    parser.add_argument('--transpose', '-t', action='store_true',
                        help='Transpose image before computing predictions')
    
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('flightline',       help="Filepath to flightline img")


    args = parser.parse_args()

    
    # Initial model setup/loading
    print("Loading CNN model...")
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
    
    print(f'using device: {device}')
    model = googlenet(pretrained=False, num_classes=2, init_weights=False)
    with open('googlenet_layers.txt','w') as fid:
        layers = list(model.children())
        for li,layer in enumerate(layers):
            lilab = f'### Layer {li+1:3.0f} of {len(layers):3.0f} '
            print(lilab + '#'*(80-len(lilab)),file=fid)
            print(str(layer),file=fid)
        print('#'*80,file=fid)

    print("[INFO] Finding model weightpath.")
    weightpaths = ['.',
                   Path(__file__).parent.resolve(),
                  '/Users/bbue/Research/srcfinder_pub/cnn/']
    
    for p in weightpaths:
        weightpath = op.join(p, "models", f"{args.model}.pt")
        if op.isfile(weightpath):
            break
        
    if op.isfile(weightpath):
        print(f"[INFO] Found {weightpath}.")
    else:
        print(f"[INFO] Model not found at {weightpath}, exiting.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(weightpath,map_location=torch.device('cpu')))
    if len(args.gpus) > 1:
        # Multi-GPU
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        model = model.to(device)
    
    model.eval()

    # Transform and dataloader
    print("Setting up Dataloader...")
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
    elif args.model == "CalCh4_v8+COVID_QC+Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[115.0],
                std=[190.0]
            )]
        )

    img = rasterio.open(args.flightline)
    imgdata = img.read([img.count])
    nodata = img.read_masks(img.count)==0

    tpstr = ''
    if args.transpose:
        imgdata = imgdata.transpose(0,2,1)
        nodata = nodata.transpose(1,0)
        tpstr = '_tpose'

    dataloader = torch.utils.data.DataLoader(
        ImageTiler(
            imgdata,
            step=args.step,
            transform=transform,
        ),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers
    )

    dim=256
    outdata = np.ones([imgdata.shape[1]+dim,imgdata.shape[2]+dim],
                      dtype=np.float32)*np.nan

    # Run shift predictions, insert into salience map at appropriate coords
    for batch,coords in tqdm(dataloader, desc="Predicting shifts"):
        inputs = batch.to(device)
        with torch.no_grad():
            preds = torch.nn.functional.softmax(model(inputs), dim=1)
            preds = preds.cpu().detach().numpy()[:,1]
            i,j = coords.cpu().detach().numpy().T
            i,j = i.squeeze(),j.squeeze()
            print(f'np.c_[i,j]: {np.c_[i,j].T}')
            outdata[i,j] = preds

    pngf = f"{Path(args.flightline).stem}-conv{tpstr}.png"
    outf = f"{Path(args.flightline).stem}-conv{tpstr}.npy"
    filf = f"{Path(args.flightline).stem}-conv{tpstr}_filled.npy"
    predf = f"{Path(args.flightline).stem}-conv{tpstr}_pred.npy"
    
    np.save(outf,outdata)
    print(f'saved: {outf}')

    # CNN input locally connected with input dim x dim tile size; 
    # downsamples dim x dim tile to stride x stride tile, where
    # stride = dim // downsampling_factor
    # FCN assumes each stride x stride patch in dim x dim receptive field
    # maps to a 1x1 cell in downsampled salience map
    # if dim == downsampling_factor, cnn_pred_pipeline == fcn_pred_pipeline
    # if dim > downsampling_factor, we assume
    # each nonoverlapping stride x stride patch in salience map will
    # share the same predicted value as its corresponding 1x1 downsampled cell
    # so we just populate empty salience grid locations via interpolation 
    mask = np.isfinite(outdata)
    j,i = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
    
    gridkw = dict(method='linear',fill_value=np.nan,rescale=True)
    print(f'Interpolating {(~mask).sum()} of {mask.size} points')
    filldata = griddata((j[mask],i[mask]),outdata[mask],(j,i),**gridkw)

    np.save(filf,filldata)      
    print(f'saved: {filf}')

    nrows,ncols = nodata.shape
    # crop to image bounds, mask nodata regions
    predimg = filldata[:nrows,:ncols]
    predimg[nodata] = np.nan
    np.save(predf,predimg)
    print(f'saved: {predf}')
    #outdata = outdata[:imgdata.shape[1],:imgdata.shape[1]]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,8))
    ax1.imshow(imgdata.squeeze(), vmin=0, vmax=4000)
    ax2.imshow(outdata, vmin=0, vmax=1.0)
    fig.savefig(pngf)
    print(f'saved: {pngf}')
        

    print("Done!")
