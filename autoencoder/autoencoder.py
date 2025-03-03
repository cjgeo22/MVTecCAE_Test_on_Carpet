

import os
import datetime
import json
import logging

import tensorflow as tf
from tensorflow import keras
import ktrain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model, metrics, and losses
from autoencoder.models import mvtecCAE
from autoencoder import metrics
from autoencoder import losses

# Configuration file for LR and callbacks (assumes you have config.py)
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoEncoder:
    def __init__(self, input_directory, architecture, color_mode, batch_size=8, verbose=True):
        self.input_directory = input_directory
        self.save_dir = None
        self.log_dir = None

        if architecture != "mvtecCAE":
            raise ValueError("Only 'mvtecCAE' architecture is supported.")
        self.architecture = architecture

        self.color_mode = color_mode
        self.loss = "mssim"  # Hard-coded for mssim
        self.batch_size = batch_size

        # Attributes for LR finder
        self.lr_opt = None
        self.lr_opt_i = None
        self.lr_base = None
        self.lr_base_i = None
        self.lr_mg_i = None
        self.lr_mg = None
        self.lr_ml_10_i = None
        self.lr_ml_10 = None

        # Training state
        self.learner = None
        self.hist = None

        # Build the mvtecCAE model
        self.model = mvtecCAE.build_model(color_mode)
        self.rescale = mvtecCAE.RESCALE
        self.shape = mvtecCAE.SHAPE
        self.preprocessing_function = mvtecCAE.PREPROCESSING_FUNCTION
        self.preprocessing = mvtecCAE.PREPROCESSING
        self.vmin = mvtecCAE.VMIN
        self.vmax = mvtecCAE.VMAX
        self.dynamic_range = mvtecCAE.DYNAMIC_RANGE

        # Load config parameters for LR finder and callbacks
        self.start_lr = config.START_LR
        self.lr_max_epochs = config.LR_MAX_EPOCHS
        self.lrf_decrease_factor = config.LRF_DECREASE_FACTOR
        self.early_stopping = config.EARLY_STOPPING
        self.reduce_on_plateau = config.REDUCE_ON_PLATEAU

        self.verbose = verbose
        if verbose:
            self.model.summary()

        # Always use mssim loss
        self.loss_function = losses.mssim_loss(self.dynamic_range)
        self.metrics = [metrics.mssim_metric(self.dynamic_range)]

        # Create directories for saving the model/logs
        self.create_save_dir()

        # Compile the model with Adam
        self.model.compile(
            loss=self.loss_function,
            optimizer="adam",
            metrics=self.metrics
        )

    def create_save_dir(self):
        now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_dir = os.path.join(
            os.getcwd(),
            "saved_models",
            self.input_directory,
            self.architecture,
            self.loss,
            now
        )
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def create_model_name(self):
        # We'll remove the .hdf5 extension to avoid HDF5 warnings
        # and save in the TF native format
        best_epoch = self.get_best_epoch()
        return f"{self.architecture}_b{self.batch_size}_e{best_epoch}"

    def find_lr_opt(self, train_generator, validation_generator):
        self.learner = ktrain.get_learner(
            model=self.model,
            train_data=train_generator,
            val_data=validation_generator,
            batch_size=self.batch_size,
        )

        logger.info("Initiating learning rate finder to determine best learning rate.")
        self.learner.lr_find(
            start_lr=self.start_lr,
            lr_mult=1.01,
            max_epochs=self.lr_max_epochs,
            stop_factor=6,
            verbose=self.verbose,
            show_plot=True,
            restore_weights_only=True,
        )
        self.ktrain_lr_estimate()
        self.custom_lr_estimate()
        self.lr_find_plot(n_skip_beginning=10, n_skip_end=1, save=True)

    def ktrain_lr_estimate(self):
        # ktrain's built-in metrics
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        self.ml_i = self.learner.lr_finder.ml
        self.lr_ml_10 = lrs[self.ml_i] / 10
        logger.info(f"lr with minimum loss / 10: {self.lr_ml_10:.2E}")

        try:
            min_loss_i = np.argmin(losses)
            self.lr_ml_10_i = np.argwhere(lrs[:min_loss_i] > self.lr_ml_10)[0][0]
        except Exception:
            self.lr_ml_10_i = None

        self.lr_mg_i = self.learner.lr_finder.mg
        if self.lr_mg_i is not None:
            self.lr_mg = lrs[self.lr_mg_i]
            logger.info(f"lr with minimum numerical gradient: {self.lr_mg:.2E}")

    def custom_lr_estimate(self):
        # Custom method to pick a learning rate
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        min_loss = np.amin(losses)
        min_loss_i = np.argmin(losses)
        segment = losses[: min_loss_i + 1]
        max_loss = np.amax(segment)

        optimal_loss = max_loss - self.lrf_decrease_factor * (max_loss - min_loss)
        indices = np.argwhere(segment < optimal_loss)
        if len(indices) == 0:
            self.lr_opt_i = min_loss_i
            self.lr_opt = float(lrs[self.lr_opt_i])
            logger.warning("No losses found below 'optimal_loss'; falling back to min_loss_i.")
        else:
            self.lr_opt_i = indices[0][0]
            self.lr_opt = float(lrs[self.lr_opt_i])

        base_indices = np.argwhere(lrs[:min_loss_i] > self.lr_opt / 10)
        if len(base_indices) == 0:
            self.lr_base_i = 0
            logger.warning("No learning rates found above (lr_opt/10); falling back to index 0.")
        else:
            self.lr_base_i = base_indices[0][0]

        self.lr_base = float(lrs[self.lr_base_i])
        logger.info(f"custom base learning rate: {self.lr_base:.2E}")
        logger.info(f"custom optimal learning rate: {self.lr_opt:.2E}")
        logger.info("Learning rate finder complete.")

    def fit(self, lr_opt):
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            write_graph=True,
            update_freq="epoch"
        )
        logger.info("Run 'tensorboard --logdir=%s' in a separate terminal to monitor training." % self.log_dir)
        assert self.learner.model is self.model

        # ktrain's autofit with cyclical LR
        self.hist = self.learner.autofit(
            lr=lr_opt,
            epochs=5,
            early_stopping=self.early_stopping,
            reduce_on_plateau=self.reduce_on_plateau,
            reduce_factor=2,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            monitor="val_loss",
            checkpoint_folder=None,
            verbose=self.verbose,
            callbacks=[tensorboard_cb],
        )

    def save(self):
        # 1) Show all keys in self.hist.history
        logger.info(f"All history keys: {list(self.hist.history.keys())}")

        # 2) Save the model in TF's native format
        model_path = os.path.join(self.save_dir, self.create_model_name())
        self.model.save(model_path, save_format="tf")
        logger.info(f"Model saved to {model_path}")

        # 3) Save training plots
        self.loss_plot(save=True)
        self.lr_schedule_plot(save=True)

        # 4) Save the history as CSV
        hist_df = pd.DataFrame(self.hist.history)
        hist_csv_file = os.path.join(self.save_dir, "history.csv")
        hist_df.to_csv(hist_csv_file, index=False)
        logger.info("Training history saved as CSV file.")
        logger.info(f"Training files saved at: {self.save_dir}")

    def get_best_epoch(self):
        """
        Return the epoch index with the lowest val_loss.
        If no val_loss is found, returns 0.
        """
        if "val_loss" not in self.hist.history:
            logger.warning("No 'val_loss' found in history; returning 0 as best epoch.")
            return 0
        best_epoch = int(np.argmin(np.array(self.hist.history["val_loss"])))
        return best_epoch

    def lr_find_plot(self, n_skip_beginning=10, n_skip_end=1, save=False):
        """
        Plot the loss vs. learning rate from the LR finder.
        """
        losses = np.array(self.learner.lr_finder.losses)
        lrs = np.array(self.learner.lr_finder.lrs)
        sb = n_skip_beginning
        se = n_skip_end

        with plt.style.context("classic"):
            fig, ax = plt.subplots()
            ax.plot(lrs[sb:-se], losses[sb:-se])
            plt.xscale("log")
            plt.xlabel("Learning Rate (log scale)")
            plt.ylabel("Loss")

            # Mark base and optimal LRs
            ax.plot(lrs[self.lr_base_i], losses[self.lr_base_i],
                    marker="o", color="green", label="custom_lr_base")
            ax.plot(lrs[self.lr_opt_i], losses[self.lr_opt_i],
                    marker="o", color="red", label="custom_lr_opt")

            # Mark min_loss/10 and min_gradient if available
            if self.lr_ml_10_i is not None:
                ax.plot(lrs[self.lr_ml_10_i], losses[self.lr_ml_10_i],
                        marker="s", color="magenta", label="lr_min_loss_div_10")
            if self.lr_mg_i is not None:
                ax.plot(lrs[self.lr_mg_i], losses[self.lr_mg_i],
                        marker="s", color="blue", label="lr_min_gradient")

            title_str = f"LR Plot\nbase LR: {lrs[self.lr_base_i]:.2E}, opt LR: {lrs[self.lr_opt_i]:.2E}"
            plt.title(title_str)
            ax.legend()
            plt.show()

        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "lr_plot.png"))
            logger.info("lr_plot.png saved.")

    def lr_schedule_plot(self, save=False):
        """
        Plot the cyclical LR schedule used during training.
        """
        with plt.style.context("classic"):
            fig, _ = plt.subplots()
            self.learner.plot(plot_type="lr")
            plt.title("Cyclical Learning Rate Scheduler")
            plt.show()
        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "lr_schedule_plot.png"))
            logger.info("lr_schedule_plot.png saved.")

    def loss_plot(self, save=False):
        """
        Plot the training and validation metrics from self.hist.history.
        """
        hist_df = pd.DataFrame(self.hist.history)
        if hist_df.empty:
            logger.warning("No history data to plot.")
            return

        with plt.style.context("classic"):
            fig = hist_df.plot().get_figure()
            plt.title("Training History")
            plt.show()

        if save:
            plt.close()
            fig.savefig(os.path.join(self.save_dir, "loss_plot.png"))
            logger.info("loss_plot.png saved.")
