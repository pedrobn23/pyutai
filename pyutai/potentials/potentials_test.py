"""
Module that implements automatic test cases for TreePotential class.
"""

import TestCase, main
import 


class TreePotentialTestCase(TestCase):
    """
    Test Class for TreePotential class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        array1 = np.array([[1,1],[2,2]])
        array2 = np.array([])
        
        self.tree1 = TreePotential.from_array(array1)
        self.tree2 = TreePotentital.from_array(array2)


if __name__ == '__main__':
    unittest.main()
