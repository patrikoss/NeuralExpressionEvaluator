from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Softmax, Dropout
from src.utils.others import CATEGORIES
from src.dataset_generators.generate_symbols_on_paper_dataset import generate as gen_dataset
from sklearn.preprocessing import LabelBinarizer
import pickle


class SymbolDetector:
    def __init__(self):
        img = Input(shape=(None, None, 3))
        x = BatchNormalization()(img)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.7)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.7)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        predictions = Conv2D(CATEGORIES + 1, (1, 1), activation='relu', padding='same')(x)
        predictions = Softmax(axis=3)(predictions)

        self.label_binarizer = LabelBinarizer()
        self.model = Model(inputs=img, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.summary()

    def train(self, train_background_folder, train_symbols_folder,
              validation_background_folder, validation_symbols_folder):
        train_gen = gen_dataset(background_folder=train_background_folder,
                                symbols_folder=train_symbols_folder,
                                batch_per_class=10,
                                label_binarizer=self.label_binarizer)
        validation_gen = gen_dataset(background_folder=validation_background_folder,
                                     symbols_folder=validation_symbols_folder,
                                     batch_per_class=10,
                                     label_binarizer=self.label_binarizer)

        self.model.fit_generator(generator=train_gen, epochs=10, steps_per_epoch=200,
                                 validation_data=validation_gen, validation_steps=25)

    def predict(self, images):
        preds_conf = self.model.predict(images)
        assert len(preds_conf.shape) == 4
        ex, h, w, cls = preds_conf.shape
        preds = preds_conf.reshape(ex*h*w, cls)
        preds_labels = self.label_binarizer.inverse_transform(preds)
        preds_labels = preds_labels.reshape(ex,h,w)
        return preds_labels, preds_conf.max(axis=3)

    def save(self, model_filepath):
        with open(model_filepath, 'wb') as filemodel:
            pickle.dump(self, filemodel)

    @staticmethod
    def load(model_filepath):
        with open(model_filepath, 'rb') as filemodel:
            symbol_classifier = pickle.load(filemodel)
        return symbol_classifier

