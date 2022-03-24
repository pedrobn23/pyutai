import os
import numpy as np

from experiments import read, utils

PATH = 'networks'


def all():
    counter = -1
    for net in os.listdir(PATH):
        if net.endswith('.bif'):
            file_ = read.read(f'{PATH}/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            for cpd in cpds:
                counter += 1
                yield (counter, cpd)


def smalls(*, threshold=3000, round_=True):
    cardinalities = set()
    counter = -1
    for _, cpd in all():
        if utils.unique_values(
                cpd) < threshold and cpd.values.shape not in cardinalities:
            cardinalities.add(cpd.values.shape)
            counter += 1

            if round_:

                cpd.values = np.around(cpd.values, decimals=2)
            yield (counter, cpd)


def small_selector(counter, *, threshold=3000):
    for counter_, cpd in smalls():
        if counter_ == counter:
            return cpd


def example(*, n_variables=6):
    for cpd in all(PATH):
        if utils.unique_values(cpd) < 3000 and len(
                cpd.values.shape) == n_variables:
            yield (0, cpd)
            return


def medical():
    for net in ['alarm.bif', 'hepar2.bif', 'munin.bif', 'pathfinder.bif']:
        fullpath = os.path.join(PATH, net)
        file_ = read.read(fullpath)
        model = file_.get_model()
        cpds = model.get_cpds()
        for cpd in cpds:
            yield cpd
