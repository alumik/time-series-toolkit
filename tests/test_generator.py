import datetime
import unittest
import numpy as np
import pandas as pd

import tskit


class TestGenerator(unittest.TestCase):

    def test_generate_index_range_index_with_start_and_end(self):
        index = tskit.generators.generate_index(start=0, end=3)
        pd.testing.assert_index_equal(pd.RangeIndex(4), index)

    def test_generate_index_range_index_with_start_and_length(self):
        index = tskit.generators.generate_index(start=0, length=4)
        pd.testing.assert_index_equal(pd.RangeIndex(4), index)

    def test_generate_index_range_index_with_end_and_length(self):
        index = tskit.generators.generate_index(end=3, length=4)
        pd.testing.assert_index_equal(pd.RangeIndex(4), index)

    def test_generate_index_range_index_with_start_and_end_and_freq(self):
        index = tskit.generators.generate_index(start=0, end=6, freq=2)
        pd.testing.assert_index_equal(pd.RangeIndex(0, 8, 2), index)

    def test_generate_index_range_index_with_start_and_length_and_freq(self):
        index = tskit.generators.generate_index(start=0, length=4, freq=2)
        pd.testing.assert_index_equal(pd.RangeIndex(0, 8, 2), index)

    def test_generate_index_datetime_index_with_start_and_end(self):
        index = tskit.generators.generate_index(
            start=datetime.datetime(2000, 1, 1),
            end=datetime.datetime(2000, 1, 4),
            freq='D',
        )
        pd.testing.assert_index_equal(pd.date_range('2000-01-01', '2000-01-04'), index)

    def test_generate_index_datetime_index_with_start_and_length_and_freq(self):
        index = tskit.generators.generate_index(
            start=pd.Timestamp('2000-01-01'),
            length=4,
            freq='D',
        )
        pd.testing.assert_index_equal(pd.date_range('2000-01-01', '2000-01-04'), index)
