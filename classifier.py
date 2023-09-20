

# add-on modules

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from feature_extraction import get_features

def train(model, features, labels, epochs_n=20):
    # Pass in network model, training feature matrix and labels
    # train for specified number of epochs_n (use a low number
    # while developing your code
    pass

def test(model, corpus, test_utterances, adv_ms, len_ms):
    # execute all, print error rate and return confusion matrix
    pass


