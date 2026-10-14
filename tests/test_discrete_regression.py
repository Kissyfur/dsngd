import unittest

import numpy as np

import discrete_reference


class DiscreteRegressionTests(unittest.TestCase):
    """The discrete case must reproduce the outputs stored from the original implementation."""

    @classmethod
    def setUpClass(cls):
        cls.golden = np.load(discrete_reference.GOLDEN_FILE)
        cls.current = discrete_reference.compute()

    def test_same_outputs_as_original_discrete_implementation(self):
        self.assertEqual(sorted(self.current), sorted(self.golden.files))
        for key in self.golden.files:
            with self.subTest(key=key):
                expected, actual = self.golden[key], np.asarray(self.current[key])
                if key in discrete_reference.APPROXIMATE_KEYS or key.endswith(discrete_reference.APPROXIMATE_SUFFIX):
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)
                else:
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
