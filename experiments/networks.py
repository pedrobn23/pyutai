import os

import numpy as np

from experiments import read, utils

PATH = 'networks'

def get(net : str):
    if net.endswith('.bif'):
        fullpath = os.path.join(PATH, net)
        reader = read.read(fullpath)
        return reader.get_model()
    else:
        raise ValueError(f'Only .bif nets are accepted, got : {net}')
    
def all_():
    counter = -1
    for net in os.listdir(PATH):
        if net.endswith('.bif'):
            bayesian_network = get(net)
            cpds = bayesian_network.get_cpds()
            for cpd in cpds:
                counter += 1
                yield (counter, cpd)

def example(*, n_variables=6):
    for cpd in all_():
        if len(cpd.values.shape) == n_variables:
            return cpd

def medical():
    for net in ['alarm.bif', 'hepar2.bif', 'munin.bif', 'pathfinder.bif']:
        bayesian_network = get(net)
        cpds = bayesian_network.get_cpds()
        for cpd in cpds:
            yield cpd, net, bayesian_network
