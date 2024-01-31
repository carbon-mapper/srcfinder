import os
import os.path as op

def check_mkdir(dirpath):
    """ Create dir if it doesn't exist """
    if not op.isdir(dirpath):
        os.mkdir(dirpath)