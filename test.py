import unittest

import numpy as np

import utils

class TestDataHandlingMethods(unittest.TestCase):
    def test_sequence_to_supervised(self):
        sequence = np.arange(20)
        x, y = utils.sequence_to_supervised(sequence, 10, 5)
        np.testing.assert_array_equal(y, np.array([ 14,15,16,17,18,19 ]))
        np.testing.assert_array_equal(x, np.array([
            [0,1,2,3,4,5, 6, 7, 8, 9 ],
            [1,2,3,4,5,6, 7, 8, 9, 10],
            [2,3,4,5,6,7, 8, 9, 10,11],
            [3,4,5,6,7,8, 9, 10,11,12],
            [4,5,6,7,8,9, 10,11,12,13],
            [5,6,7,8,9,10,11,12,13,14],
        ]))

    def test_sequence_to_supervised_all(self):
        sequence = np.arange(20)
        x, y = utils.sequence_to_supervised_all(sequence, 10, 5)
        np.testing.assert_array_equal(y, np.array([
            [10, 11, 12, 13, 14],
            [11, 12, 13, 14, 15],
            [12, 13, 14, 15, 16],
            [13, 14, 15, 16, 17],
            [14, 15, 16, 17, 18],
            [15, 16, 17, 18, 19],
        ]))
        np.testing.assert_array_equal(x, np.array([
            [0,1,2,3,4,5, 6, 7, 8, 9 ],
            [1,2,3,4,5,6, 7, 8, 9, 10],
            [2,3,4,5,6,7, 8, 9, 10,11],
            [3,4,5,6,7,8, 9, 10,11,12],
            [4,5,6,7,8,9, 10,11,12,13],
            [5,6,7,8,9,10,11,12,13,14],
        ]))

if __name__ == "__main__":
    unittest.main()
