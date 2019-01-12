from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from src.utils import CATEGORIES, load_symbols_dataset, rescale_dataset
from src.loader import save_dataset
from sklearn.preprocessing import LabelBinarizer
import pickle


class SymbolDetector:

    def __init__(self):
        img = Input(shape=(32, 32, 1))
        x = Conv2D(8, (3,3), activation='relu', padding='same')(img)
        x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        predictions = Conv2D(CATEGORIES, (3, 3), activation='sigmoid', padding='same')(x)

        self.label_binarizer = LabelBinarizer()

        # self.encoder = OneHotEncoder()
        self.model = Model(inputs=img, outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train(self, symbols_train_npz_folder):
        train_dataset = load_symbols_dataset(symbols_train_npz_folder)
        trainX, trainY = train_dataset["X"], train_dataset["y"]
        trainX = rescale_dataset(trainX, 32, 32)

        trainY = trainY.reshape(-1,1)
        trainX = trainX.reshape(*trainX.shape, -1)

        trainYbin = self.label_binarizer.fit_transform(trainY)
        trainYbin = trainYbin.reshape(trainYbin.shape[0], 1,1, trainYbin.shape[1])
        self.model.fit(trainX, trainYbin,
                       epochs=2,batch_size=128,validation_split=0.01)

    def save(self, model_filepath):
        with open(model_filepath, 'wb') as filemodel:
            pickle.dump(self, filemodel)

    @staticmethod
    def load(model_filepath):
        with open(model_filepath, 'rb') as filemodel:
            symbol_classifier = pickle.load(filemodel)
        return symbol_classifier
