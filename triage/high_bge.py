'''
Brief:   Script to identify high background enhancement flight lines
Author:  Jack Lightholder
Contact: jack.a.lightholder@jpl.nasa.gov
'''
import h5py
import shutil
import hdf5plugin
import argparse
import csv

from pandas.io.pytables    import HDFStore
from warnings              import warn
from tqdm                  import tqdm
from glob                  import glob

import matplotlib.pyplot   as plt
import pandas              as pd
import numpy               as np

def hdf2df(hdffile,**kwargs):
    '''HDF5 reader helper function'''
    store = HDFStore(hdffile)
    keys = store.keys()
    if 'key' not in kwargs:
        key = keys[0]
        if len(keys)>1:
            warn('Store "%s" contains multiple keys,'
                 ' using key="%s"'%(hdffile,key))

    result = store[key]
    store.close()
    return result

def campaign_stats(rcmfstats_file):
    '''Calculate camapgin mean/std from all HDF5 flightlines'''
    rstats = hdf2df(rcmfstats_file)

    meds = rstats['med'].to_numpy()

    meds = meds[meds < 2000] # Remove CO2 lines by only keeping values under 2000 ppmm

    campaign_mean = np.nanmean(meds)
    campaign_std = np.nanstd(meds)
    
    return campaign_mean, campaign_std

def get_available_flightline_lids(rcmfstats_file):
    '''Get all available flightline LIDS in HDF5'''
    rstats = hdf2df(rcmfstats_file)
    unique_lids = list(set(rstats['lid']))
    return unique_lids

def evaluate_flightline_bge(rcmfstats_file, lid, campaign_mean, camapaign_std):
    '''Evaluate flightline mean against campaign mean/std.  Return Z-score'''
    rstats = hdf2df(rcmfstats_file)
    meds = rstats['med'].to_numpy()
    lids = rstats['lid'].to_numpy()

    column_means = []
      
    for x in range(0, len(lids)):
        if lid == lids[x]:
            column_means.append(meds[x])
    
    flightline_mean = np.nanmean(column_means)
    
    z_score = (flightline_mean - campaign_mean)/(campaign_std)
    return z_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--rcmfstats_file',  required=True,
                                             help="rcmfstats file (generated with cmf_profile.py)")

    parser.add_argument('--output_csv_file', required=True,
                                             help="CSV path to store flightline LIDS and their respective Z Score")

    args = parser.parse_args()

    fl_z_scores = []

    # Calculate campaign level mean/std bulk statistics
    campaign_mean, campaign_std = campaign_stats(args.rcmfstats_file)
    print(f"Campaign Mean: {campaign_mean:.2f}, Standard Deviation: {campaign_std:.2f}")

    # Get list of flightline LIDS in HDF5
    flightline_lids = get_available_flightline_lids(args.rcmfstats_file)
    print(f"Found {len(flightline_lids)} unique flightlines")

    # Evaluate Z score for each flightline
    for flightline_lid in tqdm(flightline_lids):
        fl_z_score = evaluate_flightline_bge(args.rcmfstats_file, flightline_lid, campaign_mean, campaign_std)
        fl_z_scores.append(fl_z_score)

    # Save flightline LIDS and corresponding Z scores to CSV file
    with open(args.output_csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Flightline LID", "Z Score"))
        writer.writerows(zip(flightline_lids, fl_z_scores))


