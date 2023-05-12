import datetime
import unittest
import numpy as np
import pandas as pd

import tskit


class TestTimeSeries(unittest.TestCase):

    def test_init_not_equal_length(self):
        with self.assertRaises(ValueError) as e:
            tskit.TimeSeries(
                index=pd.RangeIndex(5),
                values=np.arange(4),
            )
        self.assertEqual('`index` and `values` must be of equal length.', str(e.exception))

    def test_init_integer_index(self):
        ts = tskit.TimeSeries(
            index=[0, 2, 1, 3],
            values=np.arange(4) * 100,
        )
        pd.testing.assert_index_equal(pd.Index([0, 1, 2, 3]), ts.index)
        np.testing.assert_array_equal([0, 200, 100, 300], ts.values)
        ts = tskit.TimeSeries(
            index=np.arange(4),
            values=np.arange(4) * 100,
        )
        # Caution: `np.arange(4)` generates an array of dtype `int32` by default.
        pd.testing.assert_index_equal(pd.Index([0, 1, 2, 3], dtype=np.int32), ts.index)
        np.testing.assert_array_equal([0, 100, 200, 300], ts.values)

    def test_init_range_index(self):
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(4),
            values=np.arange(4) * 100,
        )
        pd.testing.assert_index_equal(pd.RangeIndex(4), ts.index)
        np.testing.assert_array_equal([0, 100, 200, 300], ts.values)

    def test_init_date_range_index(self):
        ts = tskit.TimeSeries(
            index=pd.date_range('2000-01-01', periods=4, freq='D'),
            values=np.arange(4) * 100,
        )
        pd.testing.assert_index_equal(pd.DatetimeIndex([
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 1, 2),
            datetime.datetime(2000, 1, 3),
            datetime.datetime(2000, 1, 4),
        ]), ts.index)
        np.testing.assert_array_equal([0, 100, 200, 300], ts.values)

    def test_init_datetime_index(self):
        ts = tskit.TimeSeries(
            index=pd.DatetimeIndex([
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 2),
                datetime.datetime(2000, 1, 4),
            ]),
            values=np.arange(4) * 100,
        )
        pd.testing.assert_index_equal(pd.DatetimeIndex([
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 1, 2),
            datetime.datetime(2000, 1, 3),
            datetime.datetime(2000, 1, 4),
        ]), ts.index)
        np.testing.assert_array_equal([0, 200, 100, 300], ts.values)

    def test_init_infer_freq_integer_index(self):
        ts = tskit.TimeSeries(
            index=np.arange(4),
            values=np.arange(4),
        )
        self.assertEqual(1, ts.freq)
        ts = tskit.TimeSeries(
            index=[0, 14, 4, 18],
            values=np.arange(4),
        )
        self.assertEqual(2, ts.freq)

    def test_init_infer_freq_range_index(self):
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(4),
            values=np.arange(4),
        )
        self.assertEqual(1, ts.freq)
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(0, 8, 2),
            values=np.arange(4),
        )
        self.assertEqual(2, ts.freq)

    def test_init_infer_freq_date_range_index(self):
        ts = tskit.TimeSeries(
            index=pd.date_range('2000-01-01', periods=4, freq='D'),
            values=np.arange(4),
        )
        self.assertEqual(pd.offsets.Day(1), ts.freq)
        ts = tskit.TimeSeries(
            index=pd.date_range('2000-01-01', periods=4, freq='2s'),
            values=np.arange(4),
        )
        self.assertEqual(pd.offsets.Second(2), ts.freq)

    def test_init_infer_freq_datetime_index(self):
        ts = tskit.TimeSeries(
            index=pd.DatetimeIndex([
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 4),
                datetime.datetime(2000, 1, 5),
            ]),
            values=np.arange(4),
        )
        self.assertEqual(pd.offsets.Day(1), ts.freq)

    def test_from_generators_unknown_generators(self):
        with self.assertRaises(ValueError) as e:
            tskit.TimeSeries.from_generators(['abcd', 'random_walk'])
        self.assertEqual('Unknown generators: [\'abcd\'].', str(e.exception))

    def test_from_generators_empty_generators(self):
        with self.assertRaises(ValueError) as e:
            tskit.TimeSeries.from_generators([])
        self.assertEqual('At least one generator must be specified.', str(e.exception))

    def test_from_csv_integer_index(self):
        ts = tskit.TimeSeries.from_csv('data/test_data/test_from_csv_integer_index.csv')
        pd.testing.assert_index_equal(pd.Index([0, 2, 3, 4]), ts.index)
        np.testing.assert_array_equal([0, 200, 300, 400], ts.values)

    def test_from_csv_datetime_index(self):
        tskit.TimeSeries.from_csv(
            'data/test_data/test_from_csv_datetime_index.csv',
            to_datetime=True,
            timestamp_unit='s',
        )

    def test_len(self):
        ts = tskit.TimeSeries(
            index=pd.RangeIndex(4),
            values=np.arange(4),
        )
        self.assertEqual(4, len(ts))
