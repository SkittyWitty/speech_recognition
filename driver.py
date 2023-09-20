from argparse import ArgumentParser

import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib
# add-on packages


# project imports
from library.corpus import King
from library.audio_io import read_wav
from library.timer import Timer
from feature_extraction import get_features
from architecture import get_model
from classifier import train, test






def main(args):
    # Specify how matplotlib renders (backend library)
    # MacoOS:  'MacOSX' - native MacOS
    # TkAgg - Tcl/Tk backend
    # Qt5Agg:  Qt library, must be installed, e.g. module pyside2
    matplotlib.use("TkAgg")
    plt.ion()

    king = King(args.king, group="Nutley")  # Create instance of King corpus

    # Conduct the experiment
    # Need to load in appropriate features and labels
    # train the model against the first 5 utterances of each speaker
    # test against the second 5 of each speaker
    # and compute error rate.  Use train and test functions
    # and add any others you find appropriate






if __name__ == "__main__":

    # Process command-line arguments
    parser = ArgumentParser(
        prog="Speaker Identification",
        description="Classify speech to speaker")
    parser.add_argument("-k", "--king", required=True,
                        action="store",
                        help="King corpus directory")
    parser.add_argument("-l", "--len_ms", action="store",
                        type=float, default=20,
                        help="frame length in ms")
    parser.add_argument("-a", "--adv_ms", action="store",
                        type=float, default=10,
                        help="frame  advance in ms")

    args = parser.parse_args()

    main(args)
