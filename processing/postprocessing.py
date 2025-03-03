"""
Postprocessing module for computing residual maps (resmaps) and generating inspection plots.
Updated for Python 3.12 and modern modules.
"""

import os
import time
import numpy as np
import tensorflow as tf
from processing.utils import printProgressBar
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------- Segmentation Threshold Parameters -----------------------
# For images in float format using SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# For images in float format using L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# For images in uint8 format using SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# For images in uint8 format using L2 (less effective)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1


class TensorImages:
    """
    Class to encapsulate input and predicted images along with their computed residual maps.
    Provides functions for generating inspection plots.
    """
    def __init__(self, imgs_input, imgs_pred, vmin, vmax, method, dtype="float64", filenames=None):
        """
        Initialize the TensorImages object.

        Args:
            imgs_input (np.array): Input images.
            imgs_pred (np.array): Reconstructed (predicted) images.
            vmin (float): Minimum pixel value for display.
            vmax (float): Maximum pixel value for display.
            method (str): Method for residual map computation ("l2", "ssim", "mssim").
            dtype (str): Data type to use for residual maps ("float64" or "uint8").
            filenames (list): List of filenames corresponding to the images.
        """
        # Ensure both inputs have four dimensions
        assert imgs_input.ndim == imgs_pred.ndim == 4
        # Check that dtype and method are valid
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim", "mssim"]
        self.method = method
        self.dtype = dtype
        self.vmin = vmin
        self.vmax = vmax
        self.filenames = filenames

        # If images are grayscale (channel dimension of 1), remove the extra dimension for plotting.
        if imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]
            self.cmap = "gray"
        else:
            self.cmap = None

        # Store the input and predicted images.
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        # Compute similarity scores and residual maps.
        self.scores, self.resmaps = calculate_resmaps(self.imgs_input, self.imgs_pred, method, dtype)
        # Determine the maximum threshold from the residual maps.
        self.thresh_max = np.amax(self.resmaps)

        # Set parameters for segmentation based on dtype and method.
        if dtype == "float64":
            self.vmin_resmap = 0.0
            self.vmax_resmap = 1.0
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.thresh_step = THRESH_STEP_FLOAT_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.thresh_step = THRESH_STEP_FLOAT_L2
        elif dtype == "uint8":
            self.vmin_resmap = 0
            self.vmax_resmap = 255
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.thresh_step = THRESH_STEP_UINT8_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_UINT8_L2
                self.thresh_step = THRESH_STEP_UINT8_L2

    def generate_inspection_plots(self, group, save_dir=None):
        """
        Generate inspection plots for each image in the group.

        Args:
            group (str): Label of the image group ("validation" or "test").
            save_dir (str): Directory to save the generated plots.
        """
        assert group in ["validation", "test"]
        logger.info("Generating inspection plots on " + group + " images...")
        total_files = len(self.filenames)
        printProgressBar(0, total_files, prefix="Progress:", suffix="Complete", length=80)
        for i in range(len(self.imgs_input)):
            self.plot_input_pred_resmap(index=i, group=group, save_dir=save_dir)
            time.sleep(0.1)
            printProgressBar(i + 1, total_files, prefix="Progress:", suffix="Complete", length=80)
        if save_dir is not None:
            logger.info("All generated files are saved at:\n{}".format(save_dir))

    def plot_input_pred_resmap(self, index, group, save_dir=None):
        """
        Plot a single input image, its prediction, and the corresponding residual map.

        Args:
            index (int): Index of the image.
            group (str): Label of the group ("validation" or "test").
            save_dir (str): Directory to save the plot.
        """
        assert group in ["validation", "test"]
        # Create a figure with 3 rows: input, prediction, residual map.
        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches((4, 9))

        # Plot the original input image.
        axarr[0].imshow(self.imgs_input[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        axarr[0].set_title("Input")
        axarr[0].set_axis_off()

        # Plot the reconstructed (predicted) image.
        axarr[1].imshow(self.imgs_pred[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        axarr[1].set_title("Prediction")
        axarr[1].set_axis_off()

        # Plot the residual map (difference between input and prediction).
        im = axarr[2].imshow(self.resmaps[index], cmap="inferno", vmin=self.vmin_resmap, vmax=self.vmax_resmap)
        axarr[2].set_title("Residual Map (" + self.method + ", " + self.dtype + ")\n" + f"Score = {self.scores[index]:.2E}")
        axarr[2].set_axis_off()
        fig.colorbar(im, ax=axarr[2])

        plt.suptitle(group.upper() + "\n" + self.filenames[index])

        # Save the figure if a save directory is provided.
        if save_dir is not None:
            plot_name = get_plot_name(self.filenames[index], suffix="inspection")
            fig.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig)

    def plot_image(self, plot_type, index):
        """
        Plot an individual image of a given type.

        Args:
            plot_type (str): The type of image to plot ("input", "pred", or "resmap").
            index (int): The index of the image to plot.
        """
        assert plot_type in ["input", "pred", "resmap"]
        if plot_type == "input":
            image = self.imgs_input[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "pred":
            image = self.imgs_pred[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "resmap":
            image = self.resmaps[index]
            cmap = "inferno"
            vmin = self.vmin_resmap
            vmax = self.vmax_resmap
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        fig.colorbar(im)
        plt.title(plot_type + "\n" + self.filenames[index])
        plt.show()


def get_plot_name(filename, suffix):
    """
    Generate a new filename by appending a suffix.

    Args:
        filename (str): Original filename.
        suffix (str): Suffix to append.

    Returns:
        new_filename (str): The modified filename.
    """
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    Calculate similarity scores and residual maps between input and predicted images.

    Args:
        imgs_input (np.array): Input images.
        imgs_pred (np.array): Predicted images.
        method (str): Method to use ("l2", "ssim", or "mssim").
        dtype (str): Data type for residual maps ("float64" or "uint8").

    Returns:
        scores (list): List of similarity scores.
        resmaps (np.array): Array of residual maps.
    """
    # If images are RGB, convert them to grayscale.
    if imgs_input.ndim == 4 and imgs_input.shape[-1] == 3:
        imgs_input_gray = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
        imgs_pred_gray = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred

    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    """
    Compute residual maps using the SSIM metric.

    Args:
        imgs_input (np.array): Grayscale input images.
        imgs_pred (np.array): Grayscale predicted images.

    Returns:
        scores (list): List of SSIM scores.
        resmaps (np.array): Residual maps computed as (1 - SSIM).
    """
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        # Compute SSIM and full residual map
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    """
    Compute residual maps using the L2 (squared difference) metric.

    Args:
        imgs_input (np.array): Grayscale input images.
        imgs_pred (np.array): Grayscale predicted images.

    Returns:
        scores (list): List of L2 scores.
        resmaps (np.array): Residual maps computed as squared differences.
    """
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps


def label_images(images_th):
    """
    Label connected components in binary thresholded residual maps.

    Args:
        images_th (np.array): Array of thresholded (binary) residual maps.

    Returns:
        images_labeled (np.array): Array of labeled images.
        areas_all (list): List of lists containing areas of connected components per image.
    """
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):
        # Remove artifacts connected to the image border.
        cleared = clear_border(image_th)
        # Label connected regions.
        image_labeled = label(cleared)
        images_labeled[i] = image_labeled
        # Compute area for each region.
        regions = regionprops(image_labeled)
        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])
    return images_labeled, areas_all
