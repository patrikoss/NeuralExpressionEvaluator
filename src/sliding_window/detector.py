from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from src.utils.others import CATEGORIES
from src.dataset_generators.generate_gray_symbols_dataset import generate_gray_symbols_with_background
from sklearn.preprocessing import LabelBinarizer
import pickle


class SymbolDetector:
    # 8,8,16,16,32
    def __init__(self):
        img = Input(shape=(None, None, 3))
        x = Conv2D(16, (3,3), activation='relu', padding='same')(img)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        predictions = Conv2D(CATEGORIES+1, (1, 1), activation='softmax', padding='same')(x)

        self.label_binarizer = LabelBinarizer()

        # self.encoder = OneHotEncoder()
        self.model = Model(inputs=img, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.summary()

    def train(self, train_background_folder, train_symbols_folder, validation_background_folder, validation_symbols_folder):
        train_gen = generate_gray_symbols_with_background(background_folder=train_background_folder,
                                                          symbols_folder=train_symbols_folder,
                                                          batch_per_class=10,
                                                          label_binarizer=self.label_binarizer)
        validation_gen =generate_gray_symbols_with_background(background_folder=validation_background_folder,
                                                              symbols_folder=validation_symbols_folder,
                                                              batch_per_class=10,
                                                              label_binarizer=self.label_binarizer)

        self.model.fit_generator(generator=train_gen, epochs=5, steps_per_epoch=100,
                                 validation_data=validation_gen, validation_steps=25)

    def predict(self, images):
        preds_conf = self.model.predict(images)
        assert len(preds_conf.shape) == 4
        ex, h, w, cls = preds_conf.shape
        preds = preds_conf.reshape(ex*h*w, cls)
        preds_labels = self.label_binarizer.inverse_transform(preds)
        preds_labels = preds_labels.reshape(ex,h,w)
        #preds_conf = preds_conf / preds_conf.sum(axis=3, keepdims=True)
        return preds_labels, preds_conf.max(axis=3)


    def save(self, model_filepath):
        with open(model_filepath, 'wb') as filemodel:
            pickle.dump(self, filemodel)

    @staticmethod
    def load(model_filepath):
        with open(model_filepath, 'rb') as filemodel:
            symbol_classifier = pickle.load(filemodel)
        return symbol_classifier


"""
How to train?
train_background_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/background_color"
train_symbols_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/train/raw_symbols/npz_symbols"
validation_background_folder = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/dev/background_color"
validation_symbols_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/dev/raw_symbols/npz_symbols"
s = SymbolDetector()
s.train(...)
s.save(...)
"""