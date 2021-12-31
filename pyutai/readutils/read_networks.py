import collections
import logging
import os
import pprint

from pgmpy import readwrite
from pgmpy import models


def read_uai(path: str) -> models.BayesianNetwork:
    try:
        reader = readwrite.UAIReader(path=path)
    except OSError as ose:
        raise ValueError('Error ocurred reading network file') from ose

    if (net_type := reader.get_network_type()) != 'BAYESIAN':
        raise ValueError(
            f'Only networks of type BAYES are allowed, network in {path} is {net_type}.'
        )
    else:
        print('Look at me Pedrooooo!!')

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
    for path in classify_networks("../../networks")['BAYES']:
        read_uai(path)
