import unittest
from forecast import Forecast


class TestForecast(unittest.TestCase):
    def test_init(self):
        forecast = Forecast()
        self.assertEqual(forecast.df_sampled.values.shape, (333109, 20))
        self.assertEqual(forecast.df_filtered.values.shape, (333109, 18))
        self.assertEqual(forecast.df.values.shape, (333109, 20))


if __name__ == '__main__':
    unittest.main()
