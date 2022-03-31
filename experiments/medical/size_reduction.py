"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import dataclasses
import itertools
import json
import os
import statistics
import time

from typing import List

import numpy as np
import pandas as pd
import scipy

from pgmpy import inference
from pgmpy.factors.discrete import CPD

from potentials import cluster, element, indexpairs, indexmap, reductions, valuegrains
from potentials import utils as size_utils
from pyutai import trees

from experiments import networks, statistics
from experiments.medical import aproximation


@dataclasses.dataclass
class Result:
    original_size: int
    reduced_size: int
    cls: str
    cpd: str
    error: float
    time: float
    net: str = ''
    var: str = ''
    modified: bool = True
    cardinality: int = 0

    improvement: float = dataclasses.field(init=False)

    def __post_init__(self):
        if self.original_size != 0:
            self.improvement = 1 - self.reduced_size / self.original_size
        else:
            self.improvement = 0

    @classmethod
    def from_dict(cls, dict_: dict):
        result = cls(0, 0, object, '', 0, 0)

        for field_ in dataclasses.fields(cls):
            try:
                setattr(result, field_.name, dict_[field_.name])
            except KeyError:
                pass
        result.__post_init__()

        return result

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)


def _cpd_name(cpd: CPD.TabularCPD) -> str:
    variable = cpd.variable

    conditionals = list(cpd.variables)
    conditionals.remove(variable)

    return f'CPD in {variable} conditional on {conditionals}'


def _total_cardinality(cpd: CPD.TabularCPD) -> int:
    return np.prod(cpd.cardinality)


class _PrunedTree:
    """Auxiliar class to implement a pruned tree creator that confort standard API."""

    def __init__(self):
        pass

    @classmethod
    def from_array(cls, original_values, variables):
        tree = trees.Tree.from_array(original_values, variables)
        tree.prune()

        return tree


def reduction_size(cls: type, cpd: CPD.TabularCPD, original_values: np.ndarray,
                   reduced_values: np.ndarray, modified):

    if not modified:  # if no reduction weas carried out (both arrays are equal)
        original = cls.from_array(original_values, cpd.variables)
        reduced = original

        original_size = size_utils.total_size(original)
        reduced_size = original_size

    else:
        original = cls.from_array(original_values, cpd.variables)
        original_size = size_utils.total_size(original)

        reduced = cls.from_array(reduced_values, cpd.variables)
        reduced_size = size_utils.total_size(reduced)

    if INTERACTIVE:
        print(f'- size results for {cls} class')
        print(f'    - Original class size: {original_size}')
        print(f'    - Reduced class size: {reduced_size}')
        print(
            f'    - Total improvement: {1 - (reduced_size/original_size):.2f}% '
        )

    return original_size, reduced_size


def size_experiment(errors):
    for cpd, net_name, net in networks.medical():
        results = statistics.Statistics()

        for error in errors:
            if INTERACTIVE:
                print(
                    f'\n\n*** Results for {_cpd_name(cpd)} in net {net_name}. ***\n'
                )

            original_values, reduced_values, time_, modified = aproximation.aproximate_cpd(
                cpd, error, interactive=INTERACTIVE, verbosy = VERBOSY)

            for cls in [
                    trees.Tree, _PrunedTree, cluster.Cluster,
                    valuegrains.ValueGrains, indexpairs.IndexPairs,
                    indexmap.IndexMap
            ]:

                original_size, reduced_size = reduction_size(
                    cls, cpd, original_values, reduced_values, modified)

                results.add(
                    Result(
                        cpd=_cpd_name(cpd),
                        cls=cls.__name__,
                        error=error,
                        original_size=original_size,
                        reduced_size=reduced_size,
                        time=time_,
                        net=net_name,
                        var=cpd.variable,
                        modified=modified,
                        cardinality=_total_cardinality(cpd),
                    ))

    return results


INTERACTIVE = True
VERBOSY = False
RESULT_FILE = 'resultados_provisionales/size_results.json'
ERRORS = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
if __name__ == '__main__':

    final_results = statistics.Statistics()
    with multiprocessing.Pool(processes=len(ERRORS)) as pool:
        for results in pool.imap_unordered(size_experiment, ERRORS):
            final_results += results

    with open(RESULT_FILE, 'w') as file:
        file.write(final_results.dumps())

        
