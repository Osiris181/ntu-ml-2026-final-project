"""Checks for E11's three-outcome expected-return arithmetic."""

import unittest

import numpy as np

from members.zhenci.e11.train_e11 import expected_return


class ExpectedReturnTests(unittest.TestCase):
    def test_flat_probability_removes_loss_probability(self):
        # 40% up, 30% flat, 30% down: .4*.02 - .3*.01 = .005.
        value = expected_return(np.array([.4]), np.array([.5]),
                                np.array([.02]), np.array([.01]))
        self.assertAlmostEqual(float(value[0]), .005)

    def test_zero_expected_return_predicts_down(self):
        value = expected_return(np.array([.5]), np.array([0.]),
                                np.array([.01]), np.array([.01]))
        self.assertEqual(int((value > 0)[0]), 0)

    def test_rejects_invalid_component(self):
        with self.assertRaises(ValueError):
            expected_return(np.array([.4]), np.array([1.1]),
                            np.array([.02]), np.array([.01]))


if __name__ == "__main__":
    unittest.main()
