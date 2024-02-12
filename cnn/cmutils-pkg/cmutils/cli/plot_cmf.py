#!/usr/bin/env python
import sys, os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from skimage.io import imread

# CMF label image constants
POINTSRC = 1 # point source plume
DIFFSRC  = 2 # diffuse source plume
FALSESRC = 3 # labeled false plume
POSRGB   = (255,   0,  0)
NEGRGB   = (  0, 255,255)

def main():
    parser = argparse.ArgumentParser(os.path.split(__file__)[1])

    # positional arguments 
    parser.add_argument('cmfpath', type=str, help='CMF image path')

    # keyword arguments
    parser.add_argument('--minppmm', type=float, default=250, help='Minimum CH4 enhancement (ppmm). Default 0')
    parser.add_argument('--maxppmm', type=float, default=1500, help='Maximum CH4 enhancement (ppmm). Default 1500')    
    parser.add_argument('--maxrdn', type=float, default=15, help='Maximum radiance for scaling rgb channels. Default 15')
    parser.add_argument('--labpath', type=str, help='Override CMF label image path. Default looks for *.png or *_mask.png')
    parser.add_argument('--rgbpath', type=str, help='Override CMF rgb image path. Defaults to first 3 channels of CMF if available, otherwise zeros.')
    parser.add_argument('--outdir', type=str, default='./', help='Output directory for quicklook PDF')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI show for server CLI use')

    args = parser.parse_args()
    
    cmfpath = args.cmfpath
    cmffile = Path(cmfpath).name
    rgbpath = args.rgbpath
    
    # Read data
    print(f"Reading {cmfpath}")
    ds = rio.open(cmfpath)
    try:
        lng,lat = ds.lnglat()
        llstr = f' @ ({lat:.6f},{lng:.6f})'
    except Exception as e:
        llstr = ''
        pass
    
    dat = ds.read().transpose((1,2,0))
    NODATA = ds.nodata
    msk = (dat != NODATA).all(axis=-1)
    
    # Find and read label
    lab = np.zeros([dat.shape[0],dat.shape[1],4],dtype=np.uint8)
    labpath,labfile = None,None
    if args.labpath:
        labpath = args.labpath
    else:
        testpath1 = Path(args.cmfpath).with_suffix('.png')
        testpath2 = os.path.join(Path(args.cmfpath).parent, f"{Path(args.cmfpath).stem}_mask.png")
        if os.path.exists(testpath1):
            labpath = testpath1
        elif os.path.exists(testpath2):
            labpath = testpath2

    npos,nneg = 0,0
    if labpath:
        print(f"Reading {labpath}")
        labv = imread(labpath)
        posmsk = np.isin(labv,[POINTSRC,DIFFSRC])
        negmsk = labv==FALSESRC
        npos = np.count_nonzero(posmsk)
        nneg = np.count_nonzero(negmsk)
        lab[posmsk,:-1] = POSRGB
        lab[negmsk,:-1] = NEGRGB
        lab[labv!=0,-1] = 255
        labfile = Path(labpath).name
    
    if rgbpath:
        rgb = imread(rgbpath)
    elif dat.shape[-1]==4:
        rgb = np.clip(dat[...,:-1]/args.maxrdn,0,1)
    else:
        rgb = np.zeros([dat.shape[0],dat.shape[1],3])
        
    cmf = np.clip(dat[...,-1],args.minppmm,args.maxppmm)
    cmf[~msk | (cmf<=args.minppmm)] = np.nan

    # Generate plots
    print("Plotting...")
    figrows,figcols,figscale=1,2,5
    figsize=(figcols*figscale,figrows*figscale*1.05)
    fig,ax = plt.subplots(figrows,figcols,figsize=figsize,
                         sharex=True,sharey=True)
    ax[0].imshow(rgb)
    ax[0].imshow(cmf,vmin=args.minppmm,vmax=args.maxppmm,cmap='YlOrRd',
                 interpolation='nearest')
    ax[0].set_title(f'CMF $\in$ {[args.minppmm,args.maxppmm]}',
                    fontsize='x-small')
    

    ax[1].imshow(rgb)
    ax[1].imshow(lab,interpolation='nearest')
    ax[1].set_title(f'Labels: pos={npos}px, neg={nneg}px',
                    fontsize='x-small')

    plt.suptitle(cmffile+llstr,x=0.5,y=0.99,fontsize='x-small')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95,hspace=0.05,wspace=0.05)
    
    if not args.no_gui:
        plt.show()
    outpath = os.path.join(args.outdir, f"{Path(args.cmfpath).stem}_quicklook.pdf")
    print(f"Saving to {outpath}")
    fig.savefig(outpath)
    plt.close(fig)

if __name__ == '__main__':
    main()
