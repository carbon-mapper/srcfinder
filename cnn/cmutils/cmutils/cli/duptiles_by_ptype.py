import sys
import os
import os.path as op
import shutil
from glob import glob
from pathlib import Path
import argparse
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool
import subprocess

def _mp_tile(config):

    # Create output directories
    outdir = op.join(config['outdir'],
                     Path(config['new_flightline']).stem,
                     config['class'])
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Parse crop parameters
    # Stored numbers are 0-indexed, GDAL is 1-indexed
    x = str(int(config['suffix'].split('+')[1]))
    y = str(int(config['suffix'].split('+')[2]))
    tilesize = config['suffix'].split('+')[0].split('x')[1]
    new_filename = f"{Path(config['new_flightline']).stem}_{config['suffix']}.tif"

    # gdal_translate the crop
    subprocess.run(['gdal_translate',
                    '-of', 'GTiff',
                    '-srcwin', x, y, tilesize, tilesize,
                    config['new_flightline'],
                    op.join(outdir, new_filename)],
                    stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description='Duplicate tiles with new flightlines')
    parser.add_argument('tileset', type=str, help='Path to original tiled dataset')
    parser.add_argument('flset', type=str, help='Path to where new flightline products are located')
    parser.add_argument('ptype', type=str, help='Product type (e.g., _ch4mf_).')
    parser.add_argument('outdir', type=str, help='Output directory. Must not exist.')
    parser.add_argument('--flext', type=str, help='Expected file extension of new flightline files. None by default.', default=None)
    parser.add_argument('--cores', '--c', type=int, help='Number of parallel jobs', default=multiprocessing.cpu_count())
    args = parser.parse_args()
    ptype = args.ptype

    # Get list of flightlines in tileset
    # Looks for folders in the tileset starting with "ang"
    fls = glob(op.join(args.tileset, 'ang*/'))
    fls = set([Path(f).stem.split('_')[0] for f in fls])
    print(f"[INFO] Found {len(fls)} flightlines in {args.tileset}")

    # Get list of new flightlines in flset
    # Looks for any files in the tileset starting with "ang"
    new_fls = glob(op.join(args.flset, f'ang*{ptype}*'))
    new_fls = set([Path(f).stem.split('_')[0] for f in new_fls])
    print(f"[INFO] Found {len(new_fls)} flightlines in {args.flset}")

    # fls must be a subset of new_fls
    if not fls.issubset(new_fls):
        print(f"[WARN] Found flightlines in tileset not in flset, skip.")
        for fl in fls.difference(new_fls):
            print(fl)
        #sys.exit(1)

    # Setup outdir
    os.mkdir(args.outdir)

    ## Start building configs

    # Get list of all existing tiles
    tiles = glob(op.join(args.tileset, 'ang*', '*', '*.tif'))
    configs = []
    print(f"[INFO] Building configs")
    for t in tqdm(tiles):
        c = {}
        # Information for output directory
        c['in_flightline'] = Path(t).parent.parent.name
        c['class'] = Path(t).parent.name

        # Look for flightline to crop from
        flid = c['in_flightline'].split('_')[0]

        if flid in fls.difference(new_fls):
            continue
        if args.flext is None:
            # Expect no extension at all
            tmp = glob(op.join(args.flset, f"{flid}*{ptype}*_img"))[0]
            c['new_flightline'] = op.splitext(tmp)[0]
        else:
            c['new_flightline'] = glob(op.join(args.flset, 
                                                f"{flid}*{ptype}*_img.{args.flext}"))[0]
        c['new_flightline']
        # Information for tile filename
        c['suffix'] = Path(t).stem.split('_')[-1]
        c['outdir'] = args.outdir
        configs.append(c)

    # Multipool run
    with Pool(args.cores) as p:
        _ = list(tqdm(p.imap(_mp_tile, configs), total=len(configs)))

if __name__ == "__main__":
    main()