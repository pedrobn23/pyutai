"""
Module that implements unittests cases for PyUTAI module.
"""

import os
import unittest
import sys


def suite():
    pattern = '*_test.py'
    folder = os.path.dirname(__file__)
    return unittest.TestLoader().discover(folder, pattern=pattern)


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())  # Nice verbosy output

    result = unittest.TestResult()
    suite().run(result)
    sys.exit(len(result.errors) + len(result.failures))  # Correct exitcode
