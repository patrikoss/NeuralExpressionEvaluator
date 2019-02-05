import os
import numpy as np
from os.path import join as pjoin
from src.utils.others import SYMBOLS
from src.utils.dataset import transform_images_to_npzfile, partition_dataset, \
                                save_dataset
from src.dataset_generators.generate_encoded_decoded_dataset import generate as gen_enc_dec_dataset

# structure of a root_dataset_folder
level1 = ["all", "train_debug", "dev", "test", "train"]
level2 = ["background_color", "full_paper_color", "raw_npz_symbols", "encoded_decoded_npz_symbols"]


def prepare_dataset_folder_structure(dataset_root_folder):
    for f1 in level1:
        for f2 in level2:
            os.makedirs(pjoin(dataset_root_folder,f1,f2))


def prepare_symbols_npz_files(images_root_folder, root_dataset_folder):
    """
    For each symbol loads set of .jpg files and store them in a .npz file in root_npz_folder
    """
    for symbol in SYMBOLS:
        symbol_image_folder_path = pjoin(images_root_folder, symbol)
        target_symbol_npzfile = pjoin(root_dataset_folder, "all", "raw_npz_symbols", symbol+".npz")
        transform_images_to_npzfile(symbol_image_folder_path, target_symbol_npzfile, shuffle=True)


def partition_data_in_folder(root_dataset_folder, level1folder, level2folder):
    """
    Partitions the data found in the given folder in the 'all' level
    into train,test,dev(and train_debug) versions
    """
    partition_folder = pjoin(root_dataset_folder, level1folder, level2folder)
    for filename in os.listdir(partition_folder):
        original_filepath = pjoin(partition_folder, filename)
        all_data = np.load(original_filepath)
        partitioned_data = partition_dataset(all_data['X'], all_data['y'])

        for partition_type in level1[1:]:
            save_path = pjoin(root_dataset_folder, partition_type, level2folder, filename)
            pd_X, pd_y = partitioned_data[partition_type]
            save_dataset(pd_X, pd_y, save_path)


if __name__=='__main__':
    images_root_folder = "/path/to/folder/with/other/folders/with/jpg_symbols"
    dataset_root_folder="/target/path/to/save/npz/files/in"

    prepare_dataset_folder_structure(dataset_root_folder)
    prepare_symbols_npz_files(images_root_folder, dataset_root_folder)
    partition_data_in_folder(dataset_root_folder, 'all', 'raw_npz_symbols')
    gen_enc_dec_dataset(pjoin(dataset_root_folder, 'all', 'raw_npz_symbols'),
                        pjoin(dataset_root_folder, 'all', 'encoded_decoded_npz_symbols'))
    partition_data_in_folder(dataset_root_folder, 'all', 'encoded_decoded_npz_symbols')
