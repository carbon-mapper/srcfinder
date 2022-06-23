# AVIRIS-NG COVID Flights
This is the subdirectory for the data triage for the AVIRIS-NG COVID flights in 2020.

## COVID_systematics_ID_Deliver.py

This is a Python (3.9) code to read the complete series of RCMFSTATS generated by the cmf_profile.py script, check for columwise systematics using an approach developed by Brian Bue, then compare the results to the curated systematics stored in the file COVID_sytematics.txt. Note that this Python code uses Pandas for analysis. Also, the paths will likely need to be set appropriately for the input files.

Execution: >python3 COVID_systematic_ID_Deliver.py

Expected output: See COVID_Example_Output.txt

## COVID_systematics.txt

This is a manually generated file that contains a list of flightlines and an associated metric indicating how severe the columnwise systematics are (0 = good, 2 = severe).

## COVID_Example_Output.txt

This is an example output file that shows the expected results when run against the full set of COVID flightlines.