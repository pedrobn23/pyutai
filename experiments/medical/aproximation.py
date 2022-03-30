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


def aproximate_cpd(cpd: CPD.TabularCPD, error):
    """"""
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
        raise AssertionError('This should not happen')

    if (error_ := _entropy(original_values, reduced_values)) > error:
        raise AssertionError(
            f'Obtained error {error_}, expected under threshold {error}')

    if INTERACTIVE:
        print(f'- original number of partitions: {threshold}')
        print(f'- reduced number of partitions: {n_partitions}')
        print(f'- goal error: {error}')
        if VERBOSY:
            print(' - Original values:\n', original_values, '\n')
            print(' - Reduced values:\n', reduced_values, '\n')

    return original_values, reduced_values, time_, modified
