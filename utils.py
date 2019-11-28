#!/usr/bin/python
import os,sys

def checkFileExists(f,input_name=''):
    """
    Check the existence of an input file
    :param f: string, path to a file
    :param input_name: string, a few words to describe the input type
    :return:
    """
    if not os.path.exists(f):
        sys.exit('[thePlotFMRI.py] the input "{0}" file does not exist. Check the path :\n\t{1}'.format(input_name,f))
    else:
        return f