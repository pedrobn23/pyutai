import collections
import logging
import os
import pprint
import difflib

from pgmpy import readwrite
from pgmpy import models


def is_bayesian(path: str) -> bool:

    try:
        net_file = open(path, 'rb')

        # the first line in a UAI file contains type
        net_type = net_file.readline().strip()
        ret = net_type == b'BAYES'

    except OSError as ose:
        raise ValueError(
            f'Error ocurred reading network file {path!r}') from ose

    finally:
        net_file.close()

    return ret


def read(path: str) -> models.BayesianNetwork:
    try:
        if path.endswith(".uai"):
            if not is_bayesian(path):
                raise ValueError(f'network in {path!r} is not BAYES.' +
                                 f' Only networks of type BAYES are allowed.')

            reader = readwrite.UAIReader(path=path)

        elif path.endswith(".bif"):
            reader = readwrite.BIFReader(path=path)
        else:
            raise ValueError(
                f'File extension for path {path!r} is not supported.')

    except OSError as ose:
        raise ValueError(
            f'Error ocurred reading network file {path!r}') from ose

    return reader


def classify_networks(path: str):
    nets = collections.defaultdict(list)

    for network in os.listdir(path):
        if network.endswith(".uai"):
            with open(fullpath := os.path.join(path, network)) as net_file:
                # the first line in a UAI file contains type
                net_type = net_file.readline().strip()

                nets[net_type].append(fullpath)
    return nets


if __name__ == "__main__":
    #path = '../../networks/BN_58.uai'
    path = '../../networks/asia.bif'

    try:
        reader = read(path)
        model = reader.get_model()

        for cpd in model.get_cpds():
            print(cpd)
            print(type(cpd.values))
            print(cpd.values.shape)
    except:
        logging.exception('msg')
