import unittest
import numpy as np
from src.utils.dataset import partition_dataset
from src.utils import dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_partition_dataset(self):
        def round_diff(a, b):
            """Numerical rounding error"""
            return abs(a-b) <= 1

        n = 100
        X = np.zeros((n, 32, 32, 3))
        y = np.array(["aaaaaa"]*n)
        partition  = partition_dataset(X, y)
        trainX, trainY = partition['train']
        testX, testY = partition['test']
        devX, devY = partition['dev']
        train_debugX, train_debugY = partition['train_debug']

        # make sure the dimensions match
        self.assertTrue( round_diff(trainX.shape[0], int(n * dataset.TRAIN)) )
        self.assertTrue( round_diff(devX.shape[0], int(n * dataset.DEV)) )
        self.assertTrue( round_diff(testX.shape[0], int(n * (1-dataset.TRAIN-dataset.DEV))) )
        self.assertTrue( round_diff(train_debugX.shape[0], int(n * dataset.TRAIN_DEBUG)) )

        self.assertTrue( round_diff(trainY.shape[0], int(n * dataset.TRAIN)) )
        self.assertTrue( round_diff(devY.shape[0], int(n * dataset.DEV)) )
        self.assertTrue( round_diff(testY.shape[0], int(n * (1-dataset.TRAIN-dataset.DEV))) )
        self.assertTrue( round_diff(train_debugY.shape[0], int(n * dataset.TRAIN_DEBUG)) )

    def test_rescale_dataset(self):
        n = 100
        X = np.zeros((n,32,32,3))
        rescaled = dataset.rescale_dataset(X, 45, 45)
        self.assertEqual(rescaled.shape, (n,45,45,3))

