import unittest
import penaltyblog as pb

odds = [2.7, 2.3, 4.4]


class TestImplied(unittest.TestCase):
    def test_multiplicative(self):
        normalised = pb.implied.multiplicative(odds)
        expected = [0.358738, 0.4211273, 0.2201347]

        self.assertEqual(normalised["method"], "multiplicative")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)

    def test_additive(self):
        normalised = pb.implied.additive(odds)
        expected = [0.3595618, 0.423974, 0.2164642]

        self.assertEqual(normalised["method"], "additive")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)

    def test_power(self):
        normalised = pb.implied.power(odds)
        expected = [0.3591708, 0.4237291, 0.2171001]

        self.assertEqual(normalised["method"], "power")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)

    def test_shin(self):
        normalised = pb.implied.shin(odds)
        expected = [0.3593461, 0.4232517, 0.2174022]

        self.assertEqual(normalised["method"], "shin")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)

    def test_differential_margin_weighting(self):
        normalised = pb.implied.differential_margin_weighting(odds)
        expected = [0.3595618, 0.423974, 0.2164642]

        self.assertEqual(normalised["method"], "differential_margin_weighting")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)

    def test_odds_ratio(self):
        normalised = pb.implied.odds_ratio(odds)
        expected = [0.3588103, 0.4225611, 0.2186286]

        self.assertEqual(normalised["method"], "odds_ratio")

        for a, b in zip(normalised["implied_probabilities"], expected):
            self.assertAlmostEqual(a, b, 4)


if __name__ == "__main__":
    unittest.main()