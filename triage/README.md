# Data Triage

This directory contains some simple data triage approaches (currently only for systematics) applied to both the AVIRIS-NG COVID and Permian Basin flightlines.

### High Background Enhancement Detection
high_bge.py can be used to detect flightlines with high background enhancement.  The script calculates a bulk mean and standard deviation from all flightlines.  It then compares the mean DN value for each flightline again the bulk statistics to calculate a Z score for that flightline.  Flightlines with high positive Z scores (~>1) have high background enhancement. 

The scrpit takes two arguments:
- rcmfstats_file: An HDF5 rcmfstats file produced by the cmf_stats.py script
- output_csv_file: Path to a CSV file which will be created to store the flightline LIDS and their respective Z scores

Example usage:
```bash
$ python high_bge.py --rcmfstats_file ang_rcmfstats_2015_2020.h5 --output_csv_file z_score_results.csv
```