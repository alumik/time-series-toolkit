import unittest

import numpy as np
import pandas as pd

import tskit


class TestGenerator(unittest.TestCase):

    def test_time_series_not_equal_length(self):
        with self.assertRaises(ValueError):
            tskit.TimeSeries(
                index=pd.RangeIndex(5),
                values=[1, 2, 3, 4],
            )

    def test_time_series(self):
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(4),
            values=[1, 2, 3, 4],
        )
        pd.testing.assert_index_equal(pd.RangeIndex(4), ts.index)
        np.testing.assert_array_equal([1, 2, 3, 4], ts.values)

    def test_time_series_with_range_index(self):
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(100),
            values=np.arange(100),
            name='test_time_series_with_range_index',
        )
        ts.plot()

    def test_time_series_with_datetime_index(self):
        ts = tskit.TimeSeries(
            index=pd.date_range('2000-01-01', periods=100, freq='D'),
            values=np.arange(100),
            name='test_time_series_with_datetime_index',
        )
        ts.plot()

    def test_time_series_with_integer_index(self):
        ts = tskit.TimeSeries(
            index=[1, 2, 3, 10],
            values=[1, 2, 3, 4],
            name='test_time_series_with_integer_index',
        )
        ts.plot()
