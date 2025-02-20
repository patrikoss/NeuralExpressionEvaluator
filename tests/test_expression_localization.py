import unittest
import numpy as np
from src.utils.expression import get_expressions_boxes
from src.utils.symbol import SymbolBox


class ExpressionLocalizationTest(unittest.TestCase):
    def test_get_expression_boxes(self):
        image = 255 * np.ones((480,640), dtype=np.uint8)
        image[0,0] = 0
        image[479,639] = 0
        sb1 = SymbolBox(image, 0,0,0,0,1, "bg")
        sb2 = SymbolBox(image, 479,639,479,639, 1, "bg")
        expressions = get_expressions_boxes([sb1, sb2], image)
        self.assertEqual(len(expressions), 2)

        exp1, exp2 = expressions[0], expressions[1]
        exp1_coords =  exp1.top, exp1.left, exp1.bottom, exp1.right
        exp2_coords = exp2.top, exp2.left, exp2.bottom, exp2.right
        self.assertEqual(sorted([exp1_coords, exp2_coords]), [(0,0,0,0), (479,639,479,639)])



