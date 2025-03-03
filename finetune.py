"""
Finetuning module for selecting optimal minimum area and threshold parameters
for anomaly detection based on validation and test sets.
Updated for Python 3.12 and modern modules.
"""

import os
import argparse
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from processing import utils, postprocessing
from processing.preprocessing import Preprocessor, get_preprocessing_function
from processing.postprocessing import label_images
from processing.utils import printProgressBar
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from test import predict_classes
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_largest_areas(resmaps, thresholds):
    """
    Calculate the largest connected component area for each threshold.

    Args:
        resmaps (np.array): Residual maps.
        thresholds (np.array): Array of threshold values.

    Returns:
        largest_areas (list): Largest connected component area for each threshold.
    """
    largest_areas = []
    printProgressBar(0, len(thresholds), prefix="Progress:", suffix="Complete", length=80)
    for index, threshold in enumerate(thresholds):
        # Apply threshold to residual maps.
        resmaps_th = resmaps > threshold
        # Label connected components and compute areas.
        _, areas_th = label_images(resmaps_th)
        areas_th_total = [item for sublist in areas_th for item in sublist]
        largest_area = np.amax(np.array(areas_th_total))
        largest_areas.append(largest_area)
        time.sleep(0.1)
        printProgressBar(index + 1, len(thresholds), prefix="Progress:", suffix="Complete", length=80)
    return largest_areas


def main(args):
    """
    Main function to perform finetuning by selecting the best minimum area and threshold.
    """
    model_path = args.path
    method = args.method
    dtype = args.dtype

    # ----------------------- Load Model and Configuration -----------------------
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

    preprocessing_function = get_preprocessing_function(architecture)

    # ----------------------- Preprocess Validation Images -----------------------
    preprocessor = Preprocessor(input_directory, rescale, shape, color_mode, preprocessing_function)
    validation_generator = preprocessor.get_val_generator(batch_size=nb_validation_images, shuffle=False)
    imgs_val_input = validation_generator.next()[0]
    filenames_val = validation_generator.filenames
    imgs_val_pred = model.predict(imgs_val_input)
    tensor_val = postprocessing.TensorImages(imgs_val_input, imgs_val_pred, vmin, vmax, method, dtype, filenames_val)

    # ----------------------- Preprocess Test Images for Finetuning -----------------------
    nb_test_images = preprocessor.get_total_number_test_images()
    finetuning_generator = preprocessor.get_finetuning_generator(batch_size=nb_test_images, shuffle=False)
    imgs_test_input = finetuning_generator.next()[0]
    filenames_test = finetuning_generator.filenames

    # Stratified sampling to select a representative subset for finetuning.
    assert "good" in finetuning_generator.class_indices
    index_array = finetuning_generator.index_array
    classes = finetuning_generator.classes
    _, index_array_ft, _, classes_ft = train_test_split(index_array, classes, test_size=config.FINETUNE_SPLIT, random_state=42, stratify=classes)
    good_class_i = finetuning_generator.class_indices["good"]
    y_ft_true = np.array([0 if class_i == good_class_i else 1 for class_i in classes_ft])
    imgs_ft_input = imgs_test_input[index_array_ft]
    filenames_ft = list(np.array(filenames_test)[index_array_ft])
    imgs_ft_pred = model.predict(imgs_ft_input)
    tensor_ft = postprocessing.TensorImages(imgs_ft_input, imgs_ft_pred, vmin, vmax, method, dtype, filenames_ft)

    # ----------------------- Finetuning Parameter Search -----------------------
    dict_finetune = {"min_area": [], "threshold": [], "TPR": [], "TNR": [], "FPR": [], "FNR": [], "score": []}
    min_areas = np.arange(start=config.START_MIN_AREA, stop=config.STOP_MIN_AREA, step=config.STEP_MIN_AREA)
    thresholds = np.arange(start=tensor_val.thresh_min, stop=tensor_val.thresh_max + tensor_val.thresh_step, step=tensor_val.thresh_step)

    print("Step 1/2: Computing largest anomaly areas for increasing thresholds...")
    largest_areas = calculate_largest_areas(tensor_val.resmaps, thresholds)

    print("Step 2/2: Selecting best minimum area and threshold pair for testing...")
    printProgressBar(0, len(min_areas), prefix="Progress:", suffix="Complete", length=80)
    for i, min_area in enumerate(min_areas):
        for index, largest_area in enumerate(largest_areas):
            if min_area > largest_area:
                break
        threshold = thresholds[index]
        y_ft_pred = predict_classes(tensor_ft.resmaps, min_area, threshold)
        tnr, fpr, fnr, tpr = confusion_matrix(y_ft_true, y_ft_pred, normalize="true").ravel()
        dict_finetune["min_area"].append(min_area)
        dict_finetune["threshold"].append(threshold)
        dict_finetune["TPR"].append(tpr)
        dict_finetune["TNR"].append(tnr)
        dict_finetune["FPR"].append(fpr)
        dict_finetune["FNR"].append(fnr)
        dict_finetune["score"].append((tpr + tnr) / 2)
        printProgressBar(i + 1, len(min_areas), prefix="Progress:", suffix="Complete", length=80)

    max_score_i = np.argmax(dict_finetune["score"])
    max_score = float(dict_finetune["score"][max_score_i])
    best_min_area = int(dict_finetune["min_area"][max_score_i])
    best_threshold = float(dict_finetune["threshold"][max_score_i])

    # ----------------------- Save Finetuning Results -----------------------
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    save_dir = os.path.join(os.getcwd(), "results", input_directory, architecture, loss, model_dir_name, "finetuning", f"{method}_{dtype}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    finetuning_result = {
        "best_min_area": best_min_area,
        "best_threshold": best_threshold,
        "best_score": max_score,
        "method": method,
        "dtype": dtype,
        "split": config.FINETUNE_SPLIT,
    }
    print("Finetuning results:", finetuning_result)
    with open(os.path.join(save_dir, "finetuning_result.json"), "w") as json_file:
        json.dump(finetuning_result, json_file, indent=4, sort_keys=False)
    logger.info("Finetuning results saved at {}".format(save_dir))
    plot_min_area_threshold(dict_finetune, index_best=max_score_i, save_dir=save_dir)
    plot_scores(dict_finetune, index_best=max_score_i, save_dir=save_dir)


def plot_min_area_threshold(dict_finetune, index_best=None, save_dir=None):
    """
    Plot the relationship between min_area and threshold and mark the best pair.

    Args:
        dict_finetune (dict): Dictionary containing finetuning data.
        index_best (int): Index of the best score.
        save_dir (str): Directory to save the plot.
    """
    df_finetune = pd.DataFrame.from_dict(dict_finetune)
    with plt.style.context("seaborn-darkgrid"):
        df_finetune.plot(x="min_area", y=["threshold"], figsize=(12, 8))
        if index_best is not None:
            x = dict_finetune["min_area"][index_best]
            y = dict_finetune["threshold"][index_best]
            plt.axvline(x, 0, y, linestyle="dashed", color="red", linewidth=0.5)
            plt.axhline(y, 0, x, linestyle="dashed", color="red", linewidth=0.5)
            plt.plot(x, y, markersize=10, marker="o", color="red", label="Best min_area/threshold pair")
        plt.title(f"Min_Area Threshold Plot\nBest min_area = {x}, Best threshold = {y:.4f}")
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "min_area_threshold_plot.png"))
        print("Min_area threshold plot saved at:", save_dir)
        plt.close()


def plot_scores(dict_finetune, index_best=None, save_dir=None):
    """
    Plot TPR, TNR, and overall score versus min_area.

    Args:
        dict_finetune (dict): Dictionary containing finetuning data.
        index_best (int): Index of the best score.
        save_dir (str): Directory to save the plot.
    """
    df_finetune = pd.DataFrame.from_dict(dict_finetune)
    with plt.style.context("seaborn-darkgrid"):
        df_finetune.plot(x="min_area", y=["TPR", "TNR", "score"], figsize=(12, 8))
        if index_best is not None:
            x = dict_finetune["min_area"][index_best]
            y = dict_finetune["score"][index_best]
            plt.axvline(x, linestyle="dashed", color="red", linewidth=0.5)
            plt.plot(x, y, markersize=10, marker="o", color="red", label="Best score")
        plt.title(f"Scores Plot\nBest score = {y:.2E}")
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "scores_plot.png"))
        print("Scores plot saved at:", save_dir)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine optimal min_area and threshold for anomaly detection.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to saved model")
    parser.add_argument("-m", "--method", choices=["ssim", "l2"], default="ssim", help="Method for residual map calculation")
    parser.add_argument("-t", "--dtype", choices=["float64", "uint8"], default="float64", help="Data type for processing residual maps")
    args = parser.parse_args()
    main(args)
