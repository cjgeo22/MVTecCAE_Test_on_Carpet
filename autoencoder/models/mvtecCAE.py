"""
Implementation of the MVTec CAE architecture for anomaly detection.
Inspired by:
https://github.com/cheapthrillandwine/Improving_Unsupervised_Defect_Segmentation/blob/master/Improving_AutoEncoder_Samples.ipynb

This version is updated for Python 3.12 and modern TensorFlow/Keras.
Only the mvtecCAE model is provided.
"""

# Import TensorFlow and its Keras API
import tensorflow as tf
from tensorflow import keras

# ----------------------- Preprocessing Parameters -----------------------
# RESCALE: Scale factor used to normalize image pixel values from [0,255] to [0,1]
RESCALE = 1.0 / 255
# SHAPE: Target spatial dimensions (height, width) to which images will be resized
SHAPE = (256, 256)
# PREPROCESSING_FUNCTION: Optional additional preprocessing function (unused here)
PREPROCESSING_FUNCTION = None
# PREPROCESSING: Optional additional preprocessing parameters (unused here)
PREPROCESSING = None
# VMIN: Minimum normalized pixel value (for display and metric calculations)
VMIN = 0.0
# VMAX: Maximum normalized pixel value (for display and metric calculations)
VMAX = 1.0
# DYNAMIC_RANGE: Range of pixel values (VMAX - VMIN)
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    """
    Builds the convolutional autoencoder model using the MVTec CAE architecture.

    Args:
        color_mode (str): Specifies the input image color mode; must be either "grayscale" or "rgb".

    Returns:
        model (keras.Model): The constructed Keras autoencoder model.
    """
    # -------------------------------------------------------------------------
    # Determine the number of channels based on the specified color_mode.
    if color_mode == "grayscale":
        channels = 1  # Single channel for grayscale images
    elif color_mode == "rgb":
        channels = 3  # Three channels for RGB images
    else:
        raise ValueError("Invalid color_mode. Expected 'grayscale' or 'rgb'.")

    # -------------------------------------------------------------------------
    # Define the input layer with shape (height, width, channels)
    input_img = keras.layers.Input(shape=(*SHAPE, channels))

    # ----------------------- Encoder -----------------------------------------
    # The encoder progressively reduces spatial dimensions while increasing feature complexity.
    # First convolution: 32 filters, 4x4 kernel, stride 2, ReLU activation, "same" padding.
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(input_img)
    # Second convolution: same as above.
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Third convolution: same as above.
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Fourth convolution: 32 filters, 3x3 kernel, stride 1, ReLU activation.
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Fifth convolution: Increase to 64 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Sixth convolution: 64 filters, 3x3 kernel, stride 1.
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Seventh convolution: Increase to 128 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Eighth convolution: 64 filters, 3x3 kernel, stride 1.
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Ninth convolution: 32 filters, 3x3 kernel, stride 1.
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Bottleneck convolution: Reduce to 1 filter using an 8x8 kernel (linear activation)
    encoded = keras.layers.Conv2D(1, (8, 8), strides=1, padding="same")(x)

    # ----------------------- Decoder -----------------------------------------
    # The decoder gradually upsamples and reconstructs the image.
    # First decoder convolution: 32 filters, 3x3 kernel.
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoded)
    # Second decoder convolution: 64 filters, 3x3 kernel.
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    # First upsampling: Increase spatial dimensions by a factor of 2.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Third decoder convolution: 128 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Second upsampling.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Fourth decoder convolution: 64 filters, 3x3 kernel.
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Third upsampling.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Fifth decoder convolution: 64 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Fourth upsampling.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Sixth decoder convolution: 32 filters, 3x3 kernel.
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    # Fifth upsampling.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Seventh decoder convolution: 32 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Sixth upsampling with a larger factor.
    x = keras.layers.UpSampling2D((4, 4))(x)
    # Eighth decoder convolution: 32 filters, 4x4 kernel, stride 2.
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    # Seventh upsampling.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Ninth decoder convolution: 32 filters, 8x8 kernel.
    x = keras.layers.Conv2D(32, (8, 8), activation="relu", padding="same")(x)
    # Final upsampling to match the original dimensions.
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Output layer: Convolution that maps to the original number of channels using sigmoid activation.
    decoded = keras.layers.Conv2D(channels, (8, 8), activation="sigmoid", padding="same")(x)

    # Create the model by mapping the input to the reconstructed output.
    model = keras.models.Model(input_img, decoded)

    return model
