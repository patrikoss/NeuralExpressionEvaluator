import unittest
import numpy as np
from src.symbol_localization import get_symbols_candidates_location

class SymbolLocalizationTest(unittest.TestCase):
    def test_simple_location(self):
        arr = np.array([
            [0,     0,   0],
            [0,   255, 255],
            [255, 255,   0]
        ])
        candidate_rectangles = get_symbols_candidates_location(arr)
        self.assertEqual(len(candidate_rectangles), 2)
        self.assertEqual(sorted(candidate_rectangles), [(0,0,1,2), (2,2,2,2)])

