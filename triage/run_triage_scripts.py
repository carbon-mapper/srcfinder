#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docstring for run_triage_scripts.py
Docstrings: http://www.python.org/dev/peps/pep-0257/
"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import sys, os
from dask.distributed import Client, LocalCluster

sys.path.append(os.getenv('SRCFINDER_ROOTZ'))
from srcfinder_util import *

TRIAGE_SCRIPTS=glob(pathjoin(pathsplit(__file__)[0],'cmf_triage','*.py'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('run_triage_scripts.py')

    # keyword arguments
    parser.add_argument('-v', '--verbose', action='store_true',
    			help='Verbose output') 
    parser.add_argument('--kwp', type=str, default=None, 
    			help='kwp') 

    # positional arguments 
    parser.add_argument('cmf', type=str, help='pre-ortho cmf')
    parser.add_argument('ortcmf', type=str, help='orthoprocessed cmf')

    args = parser.parse_args()
    
    verbose = args.verbose
    kwp = args.kwp
    cmf = args.cmf
    ortcmf = args.ortcmf

    lid = filename2flightid(cmf)
    assert lid == filename2flightid(ortcmf)
    
    # replace localcluster with (PBS|Slurm)Cluster as needed
    cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1,
                           processes=True, host=None, ip=None,
                           loop=None, start=None,
                           scheduler_port=0, silence_logs=30)
    client = Client(cluster)
    
    fnargs = ()
    fnkwds = dict()
    futures = [client.submit(fn,fntgt,*fnargs,**fnkwds)
              for fntgt in fntgts]
    
    results = client.gather(futures)
    
    
