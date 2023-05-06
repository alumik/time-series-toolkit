import unittest
import pandas as pd

import tskit


class TestGenerator(unittest.TestCase):

    def test_generate_range_index_with_start_and_end(self):
        index = tskit.generator.generate_index(start=0, end=5)
        self.assertEqual(list(range(6)), list(index))

    def test_generate_range_index_with_start_and_length(self):
        index = tskit.generator.generate_index(start=0, length=5)
        self.assertEqual(list(range(5)), list(index))

    def test_generate_range_index_with_end_and_length(self):
        index = tskit.generator.generate_index(end=5, length=3)
        self.assertEqual(list(range(3, 6)), list(index))

    def test_generate_range_index_with_start_and_end_and_freq(self):
        index = tskit.generator.generate_index(start=0, end=6, freq=2)
        self.assertEqual([0, 2, 4, 6], list(index))

    def test_generate_range_index_with_start_and_length_and_freq(self):
        index = tskit.generator.generate_index(start=0, length=5, freq=2)
        self.assertEqual([0, 2, 4, 6, 8], list(index))

    def test_generate_datetime_index_with_start_and_end(self):
        index = tskit.generator.generate_index(start=pd.Timestamp('2000-01-01'), end=pd.Timestamp('2000-01-05'))
        pd.testing.assert_index_equal(pd.date_range('2000-01-01', '2000-01-05', name='timestamp'), index)

    def test_generate_datetime_index_with_start_and_length_and_freq(self):
        index = tskit.generator.generate_index(start=pd.Timestamp('2000-01-01'), length=5, freq='5D')
        pd.testing.assert_index_equal(pd.date_range('2000-01-01', periods=5, freq='5D', name='timestamp'), index)

    def test_unknown_generators(self):
        with self.assertRaisesRegex(ValueError, r'Unknown generators: \[\'abcd\'\]\.'):
            tskit.TimeSeries.from_generators(['abcd', 'random_walk'])

    def test_empty_generators(self):
        with self.assertRaisesRegex(ValueError, r'At least one generator must be specified\.'):
            tskit.TimeSeries.from_generators([])
