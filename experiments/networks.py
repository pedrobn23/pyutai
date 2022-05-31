import os

import numpy as np

from experiments import read, utils

PATH = 'networks'


def _get(net: str):
    if net.endswith('.bif'):
        fullpath = os.path.join(PATH, net)
        reader = read.read(fullpath)
        return reader.get_model()
    else:
        raise ValueError(f'Only .bif nets are accepted, got : {net}')


def all():
    for net in os.listdir(PATH):
        if net.endswith('.bif'):
            bayesian_network = _get(net)
            cpds = bayesian_network.get_cpds()
            yield (net, cpds)

def medical():
    for net in ['alarm.bif', 'hepar2.bif', 'munin.bif', 'pathfinder.bif']:
        bayesian_network = _get(net)
        cpds = bayesian_network.get_cpds()
        for cpd in cpds:
            yield cpd, net, bayesian_network
