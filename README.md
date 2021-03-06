# PyUTAI

This repository contains a Python implementation for the potentials described in [[1]](#1). You can find the library in the folder pyuntai. The test files can be found in the test folder.

## Python API
This module contains the different implementations of potentials that are to be
implemented. 

Initially it will contains the class Tree, that is a wrapper class for a tree root node.

Typical usage example:

```python

# data is read from a numpy ndarray object
  data = np.array(get_data())
  variables = ['A', 'B', 'C']
  cardinality= {'A':4, 'B':3, 'C':3}

  tree = Tree.from_array(data, variables, cardinality)

  # We can perform most of the operations over tree. For example:
  tree.prune()
  tree.access({'C':1, 'B':2})
```

## Test

In the test folder we have scripts that implement test classes based on unittest. To run all unittest use:

```
python -m test
```

from this directory. If you only want to execute a particular test module, then run:

```
python -m test.my_module_name
```

## Experimentals Results
Work in progress.
>>>>>>> 120c955e2725286652e4d17a6ec92cfdee3dea0f


## References
<a id="1">[1]</a>  Gómez‐Olmedo, Manuel, et al. "Value‐based potentials: Exploiting quantitative information regularity patterns in probabilistic graphical models." International Journal of Intelligent Systems 36.11 (2021): 6913-6943.
