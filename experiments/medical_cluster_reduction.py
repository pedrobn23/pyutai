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
from pgmpy.factors.discrete import CPD

from potentials import cluster, element, indexpairs, indexmap, reductions, valuegrains
from potentials import utils as size_utils

from experiments import networks


def ordered_elements(array: np.ndarray) -> List[element.TupleElement]:
    res = [
        element.TupleElement(state=state, value=value)
        for state, value in np.ndenumerate(array)
    ]
    res.sort(key=lambda x: x.value)
    return res


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


class Statistics:

    def __init__(self):
        self.results: List[Result] = []

    @classmethod
    def from_json(cls, path):
        stats = cls()

        with open(path, 'r') as file_:
            data = file_.read()
            stats.load(data)

        return stats


    @classmethod
    def from_files(cls, *paths):
        stats = cls()

        for path in paths:          
            if not path.endswith('.json'):
                raise ValueError(f'.json file expected, got: {path}.')

            with open(path) as file_:
                data = file_.read()
                stats.load(data)

        return stats

    def add(self, result):
        self.results.append(result)
        
    def clear(self):
        self.results.clear()

    def dumps(self) -> str:
        return json.dumps([result.asdict() for result in self.results])

    def load(self, str_: str):
        self.results += [Result.from_dict(dict_) for dict_ in json.loads(str_)]
        
    def dataframe(self):
        data = [result.astuple() for result in self.results]
        vars_ = [field_.name for field_ in dataclasses.fields(Result)]
        return pd.DataFrame(data, columns=vars_)                

    def __add__(self, other):
        results = self.results + other.results

        ret = Statistics()
        ret.results = results

        return ret


    def __str__(self):
        return str(self.results)
                
INTERACTIVE = True
VERBOSY = False

if __name__ == '__main__':
    results = Statistics()

    for index, (cpd, net) in enumerate(networks.medical()):
        for error in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
            original_values = cpd.values
            ordered_elements_ = ordered_elements(original_values)
            threshold = len(np.unique(original_values))

            start = time.time()
            reduction = reductions.Reduction.from_elements(ordered_elements_,
                                                           threshold=threshold,
                                                           interactive=False)
            end = time.time()
            time_ = end - start

            n_partitions = reduction.min_partitions(error)
            if n_partitions < threshold:
                reduced_values = reduction.array(n_partitions, cpd.cardinality)
                modified = True
            else:
                reduced_values = original_values
                modified = False
                
            if n_partitions != len(np.unique(reduced_values)):
                raise AssertionError('This should no happen')

            if INTERACTIVE:
                print(f'\n\n*** Results for {_cpd_name(cpd)}. ***\n')

                if VERBOSY:
                    print('*Original values:\n', original_values, '\n')
                    print('*Reduced values:\n', reduced_values, '\n')
                    print('')


            for cls in [
                    cluster.Cluster, valuegrains.ValueGrains,
                    indexpairs.IndexPairs, indexmap.IndexMap
            ]:

                if INTERACTIVE:
                    print(f'results for {cls} class')

                original = cls.from_array(original_values, cpd.variables)
                original_size = size_utils.total_size(original)

                reduced = cls.from_array(reduced_values, cpd.variables)
                reduced_size = size_utils.total_size(reduced)

                if INTERACTIVE:
                    print(f'- number of partitions: {n_partitions}')
                    print(f'- goal error: {error}')
                    print(f'- Original class size: {original_size}')
                    print(f'- Reduced class size: {reduced_size}')
                    print(
                        f'- Total improvement: {1 - (reduced_size/original_size):.2f}% '
                    )

                results.add(Result(cpd=_cpd_name(cpd),
                                   cls=cls.__name__,
                                   error=error,
                                   original_size=original_size,
                                   reduced_size=reduced_size,
                                   time=time_,
                                   net=net,
                                   var=cpd.variable,
                                   modified=modified))
        
        if index % 100 == 99:
            filename = f'resultados_provisionales/results{index-99}-{index}.json'
            with open(filename, 'w') as file:
                file.write(results.dumps())
