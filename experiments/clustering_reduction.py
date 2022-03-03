"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import numpy as np
import os
import statistics

from pyutai import read, cluster, values, selectors


def _cpd_size(cpd):
    return np.prod(cpd.cardinality)


def _unique_values(cpd):
    unique, _ = np.unique(cpd.values, return_counts=True)
    return len(unique)


def _select_bigs(cpds):
    return max(cpds, key=_unique_values)


def _big_cpds(path):
    test_set = []
    for net in os.listdir(path):
        if net.endswith('.bif'):
            file_ = read.read(f'networks/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            test_set.append(_select_bigs(cpds))

    return test_set


def _stats(path):
    for net in os.listdir(path):
        if net.endswith('.bif'):
            file_ = read.read(f'networks/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            unique_values = statistics.mean(_unique_values(cpd) for cpd in cpds)
            max_values = max(
                ((i, _unique_values(cpd)) for i, cpd in enumerate(cpds)),
                key=lambda x: x[1])

            print(
                f'Net: {net}. Mean unique value: {unique_values:.2f}. Biggest cpd: {max_values}'
            )


def _tree_from_cpd(cpd, next_var=None):
    if next_var is None:
        pass
    else:
        # TODO: CHECK HERE
        next_var = next_var(cpd.values, cpd.variables)

    cardinality_ = dict(zip(cpd.variables, cpd.cardinality))
    tree = values.Tree.from_array(cpd.values,
                                  cpd.variables,
                                  cardinality_,
                                  next_var=next_var)
    return tree


def _select_small(cpds, *, threshold=3000):
    return [cpd for cpd in cpds if _unique_values(cpd) < threshold]


def _medium_cpd(cpds):
    for cpd in cpds:
        if len(cpd.variables) == 6:
            return cpd

    raise ValueError()


def _small_cpds(path):
    test_set = []
    for net in os.listdir(path):
        if net.endswith('.bif'):
            file_ = read.read(f'networks/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            test_set += _select_small(cpds)

        if len(test_set) > 10:
            break
    return test_set


def _all_cpds(path):
    for net in os.listdir(path):
        if net.endswith('.bif'):
            file_ = read.read(f'networks/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            for cpd in cpds:
                yield cpd


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
