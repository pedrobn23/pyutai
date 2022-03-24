"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import collections
import os
import statistics

import numpy as np

from experiments import utils, networks, read
from pyutai import selectors


def _pruning_experiments(countered_cpds):

    total_prunning = collections.defaultdict(int)
    fault_examples = []
    for counter, cpd in countered_cpds:
        current_prunning = {}

        for selector_name, selector in [('None', None),
                                        ('variance', selectors.variance),
                                        ('entropy', selectors.entropy)]:
            tree = utils.tree_from_cpd(cpd, selector)
            size = tree.size()

            tree.prune()

            size_pruned = tree.size()

            total_prunning[selector_name] += size - size_pruned
            current_prunning[selector_name] = size - size_pruned

        if current_prunning['None'] > current_prunning['entropy']:
            fault_examples.append((counter, current_prunning))
        print(f'In cpd: {counter}:\n   {current_prunning}.')
    print(total_prunning)
    print(fault_examples)
    print(len(fault_examples))


if __name__ == '__main__':
    fault_examples = [
        4, 11, 13, 23, 27, 32, 33, 54, 59, 61, 66, 70, 72, 74, 77, 78, 79, 80,
        81, 82, 83, 86, 87, 88, 92, 94, 98, 100, 102, 103, 104, 105, 106, 107,
        109, 110, 114, 131, 137, 139, 142, 144, 146, 156, 158, 161, 166, 172,
        190, 193, 194, 197, 210, 212, 217, 226, 227, 228, 232, 233, 236, 238,
        241, 244
    ]

    cpd = networks.small_selector(233)
    print(utils.unique_values(cpd))
    print(cpd.values.shape)
    for selector_name, selector in [('None', None),
                                    ('variance', selectors.variance),
                                    ('entropy', selectors.entropy)]:
        tree = utils.tree_from_cpd(cpd, selector)
        size = tree.size()

        tree.prune()

        size_pruned = tree.size()

        print(selector_name, size, size_pruned)
