import sys
import os
import os.path as op
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import rasterio
from archs.googlenet1 import googlenet
from sklearn.metrics import classification_report

def runcmd(cmd,logprefix=None,verbose=0):
    from subprocess import Popen, PIPE
    cmdstr = ' '.join(cmd) if isinstance(cmd,list) else cmd
    if verbose:
        print("running command:",cmdstr)
    cmdout = PIPE
    for rstr in ['>>','>&','>']:
        if rstr in cmdstr:
            cmdstr,cmdout = list(map(lambda s:s.strip(),cmdstr.split(rstr)))
            mode = 'w' if rstr!='>>' else 'a'
            cmdout = open(cmdout,mode)
            
    p = Popen(cmdstr.split(), stdout=cmdout, stderr=cmdout)
    out, err = p.communicate()
    retcode = p.returncode

    if cmdout != PIPE:
        cmdout.close()

    if verbose:
        print('command completed with return code "%d"'%retcode)

    if logprefix is not None:
        savecmdlogs(logprefix,out,err)
    
    return out,err,retcode

if __name__ == "__main__":
    models = ['CalCH4_v8','COVID_QC','Permian_QC','CalCh4_v8+COVID_QC+Permian_QC']

    parser = argparse.ArgumentParser(
        description = "Generate a flightline saliency map with a CNN."
    )
    choices=["COVID_QC", "CalCH4_v8", "Permian_QC", "CalCh4_v8+COVID_QC+Permian_QC"]
    parser.add_argument('--models',        help="Models to use for prediction.",
                                            nargs='+', type=str, default=models)
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=32,
                                            type=int)
    parser.add_argument('--outdir', '-o',   help="Output directory for generated saliency maps.",
                                            default=".",
                                            type=str)
    parser.add_argument('tilecsv',       help="CSV file containing relative paths to labeled tiles",
                                            type=str)
    args = parser.parse_args()

    gpus = ' '.join([str(gid) for gid in args.gpus])
    tmpl = 'python tile_pred_pipeline.py --gpus {gpus} --batch {args.batch} --outdir {args.outdir} --model {model} {args.tilecsv}'

    for model in args.models:
        assert model in choices, f'Model "{model}" invalid, valid models=[{", ".join(choices)}]'

    csvbase = op.join(args.outdir,op.splitext(op.split(args.tilecsv)[1])[0])
    df = []
    for model in args.models:
        outf = '_'.join([csvbase,model,'predictions.csv'])
        if not os.path.exists(outf):
            cmd = tmpl.format(**locals())
            print(f'running cmd: {cmd}')
            cmdout,cmderr,cmdret = runcmd(cmd)
            if cmdret!=0: 
                print(f'cmdret: ',cmdret)
                print(f'cmdout: ',cmdout)
                print(f'cmderr: ',cmderr)
                continue
            assert op.exists(outf), f'Model {model} outfile {outf} missing'
            
        outcol = 'pred_'+model
        outdf = pd.read_csv(outf).rename(columns=dict(pred=outcol))
        if len(df)!=0:
            df[outcol] = outdf[outcol].values
        else:
            df = outdf

    outf = '_'.join([csvbase,'ensemble','predictions.csv'])
    df.to_csv(outf,index=False)

    print("Done!")


