import unittest
from src.utils.expression_evaluator import *


class SymbolLocalizationTest(unittest.TestCase):
    def test_infix_postfix(self):
        infix = ['3', '+', '4', '*', '2', '/', '(', '1', '-', '5', ')', '^', '2']
        self.assertEqual(''.join(infix_to_postfix(infix)), '342*15-2^/+')

    def test_postfix_evaluate(self):
        postfix = ['12', '2', '3', '4', '*', '10', '5', '/', '+', '*', '+']
        value = evaluate_postfix(postfix)
        self.assertEqual(evaluate_postfix(postfix), 40.)

    def test_evaluate_full_exp(self):
        exp = ['-', '5', '+', '2', '*', '3']
        self.assertEqual(evaluate(exp), 1.)
