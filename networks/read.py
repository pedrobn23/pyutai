import collections
import logging
import os
import pprint
import difflib

from pgmpy import readwrite
from pgmpy import models


def _is_bayesian(path: str) -> bool:
    """Check if a .uai network is bayesian.
    A .uai network can be bayesian o markovian. This method
    check whether the network is bayesian.
    Args:
        path: path to the file to he open.
    
    Raises:
        OSError: if there is any problem opening the file in 
            the provided path.
    """
    try:
        net_file = open(path, 'rb')

        # the first line in a UAI file contains type
        net_type = net_file.readline().strip()
        ret = net_type == b'BAYES'

    except OSError as ose:
        raise OSError(f'Error ocurred reading network file {path!r}') from ose

    finally:
        net_file.close()

    return ret


def read(path: str) -> models.BayesianNetwork:
    """
    Read a bayesian network from a file.
    Read method uses pgmpy to read a bayesian network from either
    BIF file or UAI file.
    Args:
        path: path to the file to he open.
    Raises:
        OSError: if there is any problem opening the file in 
            the provided path.
        ValueError: if file provided is not supported by read.
    """

    try:
        if path.endswith(".uai"):
            if not _is_bayesian(path):
                raise ValueError(f'network in {path!r} is not BAYES.' +
                                 f' Only networks of type BAYES are allowed.')

            reader = readwrite.UAIReader(path=path)

        elif path.endswith(".bif"):
            reader = readwrite.BIFReader(path=path)
        else:
            raise ValueError(
                f'File extension for path {path!r} is not supported.')

    except OSError as ose:
        raise OSError(f'Error ocurred reading network file {path!r}') from ose

    return reader
