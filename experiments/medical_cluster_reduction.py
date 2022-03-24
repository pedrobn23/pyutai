"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import collections
import math
import os
import statistics
import sys

import numpy as np

from experiments import utils, networks, read

if __name__ == '__main__':
    fault_examples = [
        4, 11, 13, 23, 27, 32, 33, 54, 59, 61, 66, 70, 72, 74, 77, 78, 79, 80,
        81, 82, 83, 86, 87, 88, 92, 94, 98, 100, 102, 103, 104, 105, 106, 107,
        109, 110, 114, 131, 137, 139, 142, 144, 146, 156, 158, 161, 166, 172,
        190, 193, 194, 197, 210, 212, 217, 226, 227, 228, 232, 233, 236, 238,
        241, 244
    ]

    for cpd in networks.medical():
        cluster = utils.cluster_from_cpd(cpd)

        n_clusters = len(cluster.clusters)
        print(
            f'Prior to reduction: {n_clusters}, size = {sys.getsizeof(cluster)}'
        )

        cluster = cluster.reduce_cluster(math.isqrt(n_clusters))

        n_clusters = len(cluster.clusters)
        print(
            f'Successive to reduction:{n_clusters}, size = {sys.getsizeof(cluster)}'
        )
