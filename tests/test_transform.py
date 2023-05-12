import datetime
import unittest
import numpy as np
import pandas as pd

import tskit


class TestTransform(unittest.TestCase):

    def test_interpolate_integer_index(self):
        ts = tskit.TimeSeries(
            index=[0, 2, 7],
            values=np.arange(3) * 100,
        )

        ts1 = tskit.transform.interpolate(ts)
        ts2 = tskit.transform.interpolate(ts, method='constant')
        ts3 = tskit.transform.interpolate(ts, method='history', period_length=2)

        with self.assertRaisesRegex(
                ValueError,
                r'A valid `period_length` must be provided when using the `history` method. Got .*',
        ):
            tskit.transform.interpolate(ts, method='history')

        pd.testing.assert_index_equal(pd.Index([0, 1, 2, 3, 4, 5, 6, 7]), ts1.index)
        np.testing.assert_array_equal([0, 50, 100, 120, 140, 160, 180, 200], ts1.values)
        np.testing.assert_array_equal([0, 0, 100, 0, 0, 0, 0, 200], ts2.values)
        np.testing.assert_array_equal([0, np.nan, 100, np.nan, 100, np.nan, np.nan, 200], ts3.values)

    def test_interpolate_datetime_index(self):
        ts = tskit.TimeSeries(
            index=pd.DatetimeIndex([
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 4),
            ]),
            values=np.arange(3) * 100,
        )
        ts = tskit.transform.interpolate(ts)
        pd.testing.assert_index_equal(pd.DatetimeIndex([
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 1, 2),
            datetime.datetime(2000, 1, 3),
            datetime.datetime(2000, 1, 4),
        ]), ts.index)
        np.testing.assert_array_equal([100, 50, 0, 200], ts.values)
