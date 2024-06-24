import numpy as np

from pyutai import trees
from potentials import cluster


def cpd_size(cpd):
    return np.prod(cpd.cardinality)


def unique_values(cpd):
    unique, _ = np.unique(cpd.values, return_counts=True)
    return len(unique)


def stats(net):
    if not net.endswith('.bif'):
        raise ValueError('Net format not supported. Expected .bif, got {net}')

    file_ = read.read(f'networks/{net}')
    model = file_.get_model()
    cpds = model.get_cpds()
    unique_values = statistics.mean(_unique_values(cpd) for cpd in cpds)
    max_values = max(((i, _unique_values(cpd)) for i, cpd in enumerate(cpds)),
                     key=lambda x: x[1])

    print(
        f'Net: {net}. Mean unique value: {unique_values:.2f}. Biggest cpd: {max_values}'
    )


def tree_from_cpd(cpd, selector):
    if selector is None:
        pass
    else:
        selector = selector(cpd.values, cpd.variables)

    cardinality_ = dict(zip(cpd.variables, cpd.cardinality))
    return trees.Tree.from_array(cpd.values,
                                 cpd.variables,
                                 cardinality_,
                                 selector=selector)


def cluster_from_cpd(cpd):
    return cluster.Cluster.from_array(cpd.values, cpd.variables)
