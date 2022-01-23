import unittest
import sys

from test import nodes_test, values_test


def suite():
    """Suite of test to run"""
    node_tests = unittest.TestLoader().loadTestsFromModule(nodes_test)
    tree_tests = unittest.TestLoader().loadTestsFromModule(values_test)

    return unittest.TestSuite([node_tests, tree_tests])


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())  # Nice verbosy output

    result = unittest.TestResult()
    suite().run(result)
    sys.exit(len(result.errors) + len(result.failures))  # Correct exitcode
