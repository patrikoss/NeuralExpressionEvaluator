import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import sgd
from src.utils.dataset import load_dataset
from src.utils.others import CATEGORIES,INPUT_HEIGHT, INPUT_WIDTH
import pickle


class SymbolClassifier:

    def __init__(self):
        model = Sequential()
        model.add(BatchNormalization(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=CATEGORIES, activation='softmax'))
        opt = sgd(lr=0.1)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.encoder = OneHotEncoder()
        self.model = model

    def train(self, symbols_train_npz_folder):
        train_dataset = load_dataset(symbols_train_npz_folder)
        trainX, trainY = train_dataset["X"], train_dataset["y"]

        trainY = trainY.reshape(-1,1)
        trainX = trainX.reshape(*trainX.shape, -1)

        self.encoder.fit(trainY)
        self.model.fit(trainX, self.encoder.transform(trainY), epochs=2,
                       batch_size=128,validation_split=0.01)

    def evaluate(self, symbols_dev_npz_folder):
        dev_dataset = load_dataset(symbols_dev_npz_folder)
        devX, devY = dev_dataset["X"], dev_dataset["y"]

        devY = devY.reshape(-1,1)
        devX = devX.reshape(*devX.shape, -1)

        loss_and_metrics = self.model.evaluate(devX, self.encoder.transform(devY), batch_size=128)
        return {
            "loss": loss_and_metrics[0],
            "accuracy": loss_and_metrics[1]
        }

    def save(self, model_filepath):
        with open(model_filepath, 'wb') as filemodel:
            pickle.dump(self, filemodel)

    def predict_probabilties(self, np_images):
        assert len(np_images.shape) in {2, 3}
        if len(np_images.shape) == 2:
            images = np_images.reshape(1, *np_images.shape, 1)
        else:
            images = np_images.reshape(*np_images.shape, 1)
        probabilities = self.model.predict(images)
        return probabilities

    def predict(self, np_images):
        """
        Returns pair of list of predictions:
        labels, probabilitiess
        """
        probabilities = self.predict_probabilties(np_images)
        if not len(probabilities):
            return np.empty(0, 'S6')
        predictions = np.argmax(probabilities, axis=1)
        return self.encoder.categories_[0][predictions], probabilities.max(axis=1)

    @staticmethod
    def load(model_filepath):
        with open(model_filepath, 'rb') as filemodel:
            symbol_classifier = pickle.load(filemodel)
        return symbol_classifier




"""
# How to train?
sc = SymbolClassifier1("path/to/encoded_decoded_dataset")
sc.save("...")
"""
