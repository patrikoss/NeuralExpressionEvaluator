import unittest
import numpy as np
from src.symbol import get_symbols_candidates_location


class SymbolLocalizationTest(unittest.TestCase):
    def test_simple_location(self):
        arr = np.array([
            [0,     0,   0],
            [0,   255, 255],
            [255, 255,   0]
        ])
        candidate_rectangles = get_symbols_candidates_location(arr, arr)
        self.assertEqual(len(candidate_rectangles), 2)
        rec1, rec2 = candidate_rectangles[0], candidate_rectangles[1]
        rec1_cords = (rec1.top, rec1.left, rec1.bottom, rec1.right)
        rec2_cords = (rec2.top, rec2.left, rec2.bottom, rec2.right)
        self.assertEqual(sorted([rec1_cords, rec2_cords]), [(0,0,1,2), (2,2,2,2)])

