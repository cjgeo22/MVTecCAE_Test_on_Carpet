"""
train.py - Example training script for the MVTec CAE model using mssim.
"""

import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras

# Import the updated AutoEncoder class
from autoencoder.autoencoder import AutoEncoder
# Preprocessing and postprocessing modules (assuming these exist in your project)
from processing.preprocessing import Preprocessor
from processing import postprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    input_dir = args.input_dir
    architecture = args.architecture
    color_mode = args.color
    batch_size = args.batch

    # 1) Initialize the AutoEncoder (mvtecCAE only) - uses mssim by default
    autoencoder = AutoEncoder(
        input_directory=input_dir,
        architecture=architecture,
        color_mode=color_mode,
        batch_size=batch_size,
        verbose=True
    )

    # 2) Create a Preprocessor for data
    preprocessor = Preprocessor(
        input_directory=input_dir,
        rescale=autoencoder.rescale,
        shape=autoencoder.shape,
        color_mode=autoencoder.color_mode,
        preprocessing_function=autoencoder.preprocessing_function,
    )
    train_generator = preprocessor.get_train_generator(batch_size=autoencoder.batch_size, shuffle=True)
    validation_generator = preprocessor.get_val_generator(batch_size=autoencoder.batch_size, shuffle=True)

    # 3) Find the optimal learning rate
    autoencoder.find_lr_opt(train_generator, validation_generator)

    # 4) Train the model using that optimal learning rate
    autoencoder.fit(lr_opt=autoencoder.lr_opt)

    # 5) Save the trained model and training artifacts
    autoencoder.save()

    # 6) (Optional) Generate inspection plots if requested
    if args.inspect:
        logger.info("Generating inspection plots for validation images...")
        inspection_val_dir = os.path.join(autoencoder.save_dir, "inspection_val")
        os.makedirs(inspection_val_dir, exist_ok=True)

        # Load all validation images
        inspection_val_generator = preprocessor.get_val_generator(
            batch_size=autoencoder.learner.val_data.samples,
            shuffle=False
        )
        imgs_val_input = inspection_val_generator.next()[0]
        filenames_val = inspection_val_generator.filenames
        logger.info("Reconstructing validation images...")
        imgs_val_pred = autoencoder.model.predict(imgs_val_input)

        tensor_val = postprocessing.TensorImages(
            imgs_val_input,
            imgs_val_pred,
            autoencoder.vmin,
            autoencoder.vmax,
            "mssim",  # Just a label for your postprocessing
            "float64",
            filenames_val,
        )
        tensor_val.generate_inspection_plots(group="validation", save_dir=inspection_val_dir)

        logger.info("Generating inspection plots for test images...")
        inspection_test_dir = os.path.join(autoencoder.save_dir, "inspection_test")
        os.makedirs(inspection_test_dir, exist_ok=True)

        nb_test_images = preprocessor.get_total_number_test_images()
        inspection_test_generator = preprocessor.get_test_generator(batch_size=nb_test_images, shuffle=False)
        imgs_test_input = inspection_test_generator.next()[0]
        filenames_test = inspection_test_generator.filenames
        logger.info("Reconstructing test images...")
        imgs_test_pred = autoencoder.model.predict(imgs_test_input)

        tensor_test = postprocessing.TensorImages(
            imgs_test_input,
            imgs_test_pred,
            autoencoder.vmin,
            autoencoder.vmax,
            "mssim",
            "float64",
            filenames_test,
        )
        tensor_test.generate_inspection_plots(group="test", save_dir=inspection_test_dir)

    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder on an image dataset (mssim only).",
        epilog="Example: python train.py -d mvtec/capsule -a mvtecCAE -c rgb -b 8 --inspect"
    )
    parser.add_argument("-d", "--input-dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("-a", "--architecture", type=str, choices=["mvtecCAE"], default="mvtecCAE",
                        help="Model architecture (only mvtecCAE is supported)")
    parser.add_argument("-c", "--color", type=str, choices=["rgb", "grayscale"], default="rgb",
                        help="Color mode for preprocessing images")
    parser.add_argument("-b", "--batch", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("-i", "--inspect", action="store_true",
                        help="Generate inspection plots after training")
    args = parser.parse_args()

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info("GPU detected.")
    else:
        logger.info("No GPU detected. Training might be slow.")
    logger.info(f"TensorFlow version: {tf.__version__}")

    main(args)
