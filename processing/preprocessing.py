"""
Preprocessing module for preparing image data for training and inference.
Updated for Python 3.12 and modern TensorFlow.
"""

import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


class Preprocessor:
    """
    Class to create data generators for training, validation, testing, and finetuning.
    """
    def __init__(self, input_directory, rescale, shape, color_mode, preprocessing_function):
        """
        Initialize the Preprocessor.

        Args:
            input_directory (str): Base directory containing the dataset.
            rescale (float): Factor to rescale pixel values.
            shape (tuple): Target image shape (height, width).
            color_mode (str): Color mode ("rgb" or "grayscale").
            preprocessing_function (function): Optional preprocessing function.
        """
        self.input_directory = input_directory
        # Define directories for training and testing images.
        self.train_data_dir = os.path.join(input_directory, "train")
        self.test_data_dir = os.path.join(input_directory, "test")
        self.rescale = rescale
        self.shape = shape
        self.color_mode = color_mode
        self.preprocessing_function = preprocessing_function
        # Validation split parameter from configuration.
        self.validation_split = config.VAL_SPLIT

        self.nb_val_images = None
        self.nb_test_images = None

    def get_train_generator(self, batch_size, shuffle=True):
        """
        Create a generator for training data with augmentation.

        Args:
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            train_generator: Data generator for training images.
        """
        train_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=config.ROT_ANGLE,
            width_shift_range=config.W_SHIFT_RANGE,
            height_shift_range=config.H_SHIFT_RANGE,
            fill_mode=config.FILL_MODE,
            cval=0.0,
            brightness_range=config.BRIGHTNESS_RANGE,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_function,
            data_format="channels_last",
            validation_split=self.validation_split,
        )
        train_generator = train_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            subset="training",
            shuffle=shuffle,
        )
        return train_generator

    def get_val_generator(self, batch_size, shuffle=True):
        """
        Create a generator for validation data.

        Args:
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            validation_generator: Data generator for validation images.
        """
        validation_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            validation_split=self.validation_split,
            preprocessing_function=self.preprocessing_function,
        )
        validation_generator = validation_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            subset="validation",
            shuffle=shuffle,
        )
        return validation_generator

    def get_test_generator(self, batch_size, shuffle=False):
        """
        Create a generator for test data.

        Args:
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            test_generator: Data generator for test images.
        """
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )
        test_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator

    def get_finetuning_generator(self, batch_size, shuffle=False):
        """
        Create a generator for finetuning data (subset of test images).

        Args:
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            finetuning_generator: Data generator for finetuning.
        """
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )
        finetuning_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return finetuning_generator

    def get_total_number_test_images(self):
        """
        Count the total number of test images.

        Returns:
            total_number (int): Total number of images in the test directory.
        """
        total_number = 0
        sub_dir_names = os.listdir(self.test_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(self.test_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            total_number += len(filenames)
        return total_number


def get_preprocessing_function(architecture):
    """
    Return the preprocessing function based on the model architecture.
    For mvtecCAE, no additional preprocessing is needed.

    Args:
        architecture (str): Model architecture name.

    Returns:
        preprocessing_function: None (for mvtecCAE).
    """
    if architecture == "mvtecCAE":
        return None
    return None
