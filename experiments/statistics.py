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

from experiments import networks
from experiments.medical import aproximation


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
