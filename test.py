"""
Testing module to evaluate the AutoEncoder on test images.
Updated for Python 3.12 and modern modules.
"""

import sys
import os
import argparse
from pathlib import Path
import time
import json
import tensorflow as tf
from processing import utils, postprocessing
from processing.preprocessing import Preprocessor, get_preprocessing_function
from processing.postprocessing import label_images
from processing.utils import printProgressBar
from skimage.util import img_as_ubyte
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_true_classes(filenames):
    """
    Derive ground truth classes from filenames.
    Filenames not containing "good" are considered defective.

    Args:
        filenames (list): List of image filenames.

    Returns:
        y_true (list): List of true class labels (0 for good, 1 for defective).
    """
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]
    return y_true


def is_defective(areas, min_area):
    """
    Decide if an image is defective based on the areas of its connected components.

    Args:
        areas (list): Areas of connected components.
        min_area (int): Minimum area threshold.

    Returns:
        int: 1 if defective, 0 otherwise.
    """
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def predict_classes(resmaps, min_area, threshold):
    """
    Predict class labels based on thresholding residual maps.

    Args:
        resmaps (np.array): Residual maps.
        min_area (int): Minimum area threshold.
        threshold (float): Residual map threshold.

    Returns:
        y_pred (list): Predicted class labels.
    """
    resmaps_th = resmaps > threshold
    _, areas_all = label_images(resmaps_th)
    y_pred = [is_defective(areas, min_area) for areas in areas_all]
    return y_pred


def save_segmented_images(resmaps, threshold, filenames, save_dir):
    """
    Save segmented residual maps as images.

    Args:
        resmaps (np.array): Residual maps.
        threshold (float): Threshold used for segmentation.
        filenames (list): List of original filenames.
        save_dir (str): Directory to save the segmented images.
    """
    resmaps_th = resmaps > threshold
    seg_dir = os.path.join(save_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)
    for i, resmap_th in enumerate(resmaps_th):
        fname = utils.generate_new_name(filenames[i], suffix="seg")
        fpath = os.path.join(seg_dir, fname)
        plt.imsave(fpath, resmap_th, cmap="gray")


def main(args):
    """
    Main function to test the model on test images.
    """
    model_path = args.path
    save = args.save

    model, info, _ = utils.load_model_HDF5(model_path)
    input_directory = info["data"]["input_directory"]
    architecture = info["model"]["architecture"]
    loss = info["model"]["loss"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]
    nb_validation_images = info["data"]["nb_validation_images"]

    model_dir_name = os.path.basename(str(Path(model_path).parent))
    finetune_dir = os.path.join(os.getcwd(), "results", input_directory, architecture, loss, model_dir_name, "finetuning")
    subdirs = os.listdir(finetune_dir)
    for subdir in subdirs:
        logger.info("Testing with finetuning parameters from {}".format(os.path.join(finetune_dir, subdir)))
        try:
            with open(os.path.join(finetune_dir, subdir, "finetuning_result.json"), "r") as read_file:
                validation_result = json.load(read_file)
        except FileNotFoundError:
            logger.warning("Run finetune.py before testing. Exiting script.")
            sys.exit()
        min_area = validation_result["best_min_area"]
        threshold = validation_result["best_threshold"]
        method = validation_result["method"]
        dtype = validation_result["dtype"]

        preprocessing_function = get_preprocessing_function(architecture)
        preprocessor = Preprocessor(input_directory, rescale, shape, color_mode, preprocessing_function)
        nb_test_images = preprocessor.get_total_number_test_images()
        test_generator = preprocessor.get_test_generator(batch_size=nb_test_images, shuffle=False)
        imgs_test_input = test_generator.next()[0]
        filenames = test_generator.filenames
        imgs_test_pred = model.predict(imgs_test_input)
        tensor_test = postprocessing.TensorImages(imgs_test_input, imgs_test_pred, vmin, vmax, method, dtype, filenames)
        y_true = get_true_classes(filenames)
        y_pred = predict_classes(tensor_test.resmaps, min_area, threshold)
        tnr, fp, fn, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()
        test_result = {"min_area": min_area, "threshold": threshold, "TPR": tpr, "TNR": tnr, "score": (tpr + tnr) / 2, "method": method, "dtype": dtype}
        save_dir = os.path.join(os.getcwd(), "results", input_directory, architecture, loss, model_dir_name, "test", subdir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "test_result.json"), "w") as json_file:
            json.dump(test_result, json_file, indent=4, sort_keys=False)
        classification = {
            "filenames": filenames,
            "predictions": y_pred,
            "truth": y_true,
            "accurate_predictions": np.array(y_true) == np.array(y_pred),
        }
        df_clf = pd.DataFrame.from_dict(classification)
        with open(os.path.join(save_dir, "classification.txt"), "w") as f:
            f.write(f"min_area = {min_area}, threshold = {threshold}, method = {method}, dtype = {dtype}\n\n")
            f.write(df_clf.to_string(header=True, index=True))
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df_clf)
        if save:
            save_segmented_images(tensor_test.resmaps, threshold, filenames, save_dir)
        print("Test results:", test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to saved model")
    parser.add_argument("-s", "--save", action="store_true", help="Save segmented images")
    args = parser.parse_args()
    main(args)
