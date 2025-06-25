import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras_hub.layers import TransformerDecoder


def reweight_distribution(original_distribution, temperature=0.5):
    """Given a softmax probability distribution, reweight the probabilities
    using a temperature variable.  Temperature 0 = greedy sampling, temperature
    1 = original distribution, Temperature > 1, more randomness until
    completely uniform distribution.

    Parameters
    ----------
    original_distribution : ndarray 
        Vector of probability over a vocabulary, probability must sum to 1.0
    temperature : float
        Factor quantifying the entropy of the output distribution

    Returns
    -------
    new_distribution : ndarray
        A reweighted version of the original distribution.
    """
    # NOTE: multinomial issue sometimes causes sum to be > 1.0 because it casts to float64,
    # we seem to be able to avoid by casing before we recalculate the distribution
    original_distribution = original_distribution.astype(np.float64)
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    # sum may longer be 1.0, so divide by the sum to obtain new probability
    # distribution that sums to 1.0
    return distribution / np.sum(distribution)


def plot_history(ax, history_dict, metric_key):
    """Plot the asked for metrics. Usually we need to plot the metric from the training
    data and its corresponding measurement using the validation data, thus we pass in
    two keys for the training and validation metric to plot.

    Arguments
    ---------
    ax - a matplotlib figure axis to create plot onto
    history_dict - A Python dictionary whose keys should return list like enumerable
      items holding the measured metrics over some number of epochs of training.
    metric_key - The string key for the metric, validation data is assumed to be
      accessible as "val_" + metric_key


    """
    # setup epochs and keys/labels for the plot
    train_key = metric_key
    train_label = "Training " + metric_key
    val_key = "val_" + metric_key
    val_label = "Validation " + metric_key
    epochs = np.arange(1, len(history_dict[train_key]) + 1)
    
    # create the plot of the train and test metric
    ax.plot(epochs, history_dict[train_key], 'r-', label=train_label)
    if val_key in history_dict.keys():
        ax.plot(epochs, history_dict[val_key], 'b-', label=val_label)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_key)
    #ax.set_xticks(epochs)
    ax.grid()
    ax.legend();

