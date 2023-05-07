import datetime
import unittest
import numpy as np
import pandas as pd

import tskit


class TestGenerator(unittest.TestCase):

    def test_interpolate_integer_index(self):
        ts = tskit.TimeSeries(
            index=[0, 2, 7],
            values=[0, 1, 2],
            name='test_interpolate_integer_index',
        )
        ts1 = tskit.transform.interpolate(ts)
        ts2 = tskit.transform.interpolate(ts, method='constant')
        with self.assertRaisesRegex(
                ValueError,
                r'A valid `period_length` must be provided when using the `history` method. Got .*',
        ):
            tskit.transform.interpolate(ts, method='history')
        ts3 = tskit.transform.interpolate(ts, method='history', period_length=2)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7], list(ts1.index))
        np.testing.assert_array_equal([0, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2], ts1.values)
        np.testing.assert_array_equal([0, 0, 1, 0, 0, 0, 0, 2], ts2.values)
        np.testing.assert_array_equal([0, np.nan, 1, np.nan, 1, np.nan, np.nan, 2], ts3.values)

    def test_interpolate_datetime_index(self):
        ts = tskit.TimeSeries(
            index=pd.DatetimeIndex([
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 4),
            ]),
            values=[0, 1, 2],
            name='test_interpolate_datetime_index',
        )
        ts = tskit.transform.interpolate(ts)
        self.assertEqual(
            [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 2),
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 4),
            ],
            list(ts.index))
        np.testing.assert_array_equal([1, 0.5, 0, 2], ts.values)
