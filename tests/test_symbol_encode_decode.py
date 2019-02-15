import unittest
import numpy as np
from src.utils.symbol import encode_symbol, decode_symbol

class TestEncodeDecode(unittest.TestCase):
    def setUp(self):
        pass

    def test_encode_symbol(self):
        image = np.zeros((32,32))
        encoded = encode_symbol(image, 45, 45)
        self.assertEqual(encoded.shape, (45, 45))

    def test_decode_symbol(self):
        image = np.zeros((45,45))
        decoded = decode_symbol(image, 32, 32)
        self.assertEqual(decoded.shape, (32,32))
