import unittest
import penaltyblog as pb


class TestRps(unittest.TestCase):
    def test_rps(self):
        predictions = [0.8, 0.1, 0.1]
        observed = 0

        rps_score = pb.utilities.rps(predictions, observed)

        self.assertAlmostEqual(rps_score, 0.025, 4, "Should be 0.025")


if __name__ == "__main__":
    unittest.main()
