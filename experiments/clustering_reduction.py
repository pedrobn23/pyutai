"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import numpy as np
import os
import statistics

from pyutai import read, cluster, values, selectors
from experiments import utils, networks



def _tree_from_cluster(cluster_, variables):
    return values.Tree.from_callable(data=cluster_.access,
                                     variables=variables,
                                     cardinalities=cluster_.cardinalities)


def _cluster_experiment(test_set):
    differences = []

    for cpd in test_set:
        tree = _tree_from_cpd(cpd)
        variables = cpd.variables
        cluster_ = cluster.Potential.from_tree(tree)
        length = len(cluster_.clusters)

        if length > 1:
            print('\n ------ \n')
            print(f'Prior cluster size: {length}.')
            print(f'Tree prior unpruned: {tree.size()}.')

            tree.prune()
            tree1size = tree.size()
            print(f'Tree prior pruned: {tree1size}.')

            reduced_cluster = cluster_.reduce_cluster(length // 2)
            reduced_length = len(reduced_cluster.clusters)
            print(f'Post cluster size: {reduced_length}.')

            tree2 = _tree_from_cluster(reduced_cluster, variables,
                                       selectors.variance)
            tree2.prune()
            tree2size = tree2.size()

            print(f'Tree post pruned: {tree2size}.')
            print(f'    - diference: {tree1size - tree2size}.')

            differences.append(tree1size - tree2size)

    print(statistics.mean(differences))


def _pruning_experiments(path):
    for selector in [
            None, selectors.variance, selectors.entropy
    ]:
        total_pruning = 0
        test_set = _all_cpds(path)
        for cpd in test_set:
            tree = _tree_from_cpd(cpd, selector)
            size = tree.size()

            tree.prune()

            size_pruned = tree.size()

            total_pruning += size - size_pruned

        print(
            f'With selector {selector} we achieve and improvement of {total_pruning} less nodes.'
        )


def _mono_pruning_experiment(cpd):
    for selector in [
            None, selectors.variance, selectors.entropy
    ]:
        total_pruning = 0

        tree = _tree_from_cpd(cpd, selector)
        size = tree.size()

        tree.prune()

        size_pruned = tree.size()

        total_pruning += size - size_pruned

        print(
            f'With selector {selector} we achieve and improvement of {total_pruning} less nodes.'
        )


if __name__ == '__main__':

    _pruning_experiments('networks')
