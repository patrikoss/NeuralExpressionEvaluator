from sklearn.preprocessing import LabelBinarizer
from src.dataset_generators.generate_gray_symbols_dataset import generate_gray_symbols_with_background
from src.utils.others import show_images
import cv2

train_background_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/all/background_color"
train_symbols_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/train/raw_symbols/npz_symbols"
validation_background_folder = "/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/dev/background_color"
validation_symbols_folder="/home/patryk/PycharmProjects/MathExpressionEvaluator/dataset/dev/raw_symbols/npz_symbols"

label_binarizer = LabelBinarizer()
train_gen = generate_gray_symbols_with_background(background_folder=train_background_folder,
                                                  symbols_folder=train_symbols_folder,
                                                  batch_per_class=10,
                                                  label_binarizer=label_binarizer)
validation_gen = generate_gray_symbols_with_background(background_folder=validation_background_folder,
                                                       symbols_folder=validation_symbols_folder,
                                                       batch_per_class=10,
                                                       label_binarizer=label_binarizer)

X, y = next(train_gen)
X = X.astype('uint8')


#y = y.argmax(axis=3).reshape(y.shape[0])
preds_label_nrs = y.argmax(axis=3).reshape(y.shape[0])
preds_labels = label_binarizer.classes_[preds_label_nrs]
show_images(X[:25],preds_labels[:25])

