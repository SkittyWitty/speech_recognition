"""
I promise that the attached assignment is my own work. 
I recognize that should this not be the case,
I will be subject to penalties as outlined in the course syllabus.
Mindy Flores
"""

from argparse import ArgumentParser
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib

# project imports
from library.corpus import King
from library.audio_io import read_wav
from library.timer import Timer
from feature_extraction import get_features
from architecture import get_model
from classifier import train, test
from sklearn.preprocessing import OneHotEncoder

def main(args):
    # Specify how matplotlib renders (backend library)
    matplotlib.use("QtAgg")
    plt.ion()

    adv_ms = args.adv_ms
    len_ms = args.len_ms

    # Expect speakers 27 to 60 to be availible
    # Total: 25 speakers
    king = King(args.king, group="Nutley")
    
    speakers = list(king.get_speakers()) # We can use catagory_2_speaker to translate index to speaker ID
    encorder = create_one_hot(speakers)
    training_speakers = speakers[:5]
    testing_speakers = speakers[5:]

    # Training data
    train_samples = None
    train_labels  = None
    for i in training_speakers:
        for file in king[i]:
            features, labels = get_features(file, adv_ms, len_ms, i)
            labels = encorder.transform(labels.reshape(-1, 1)).A

            if(train_samples is not None):
                train_samples = np.concatenate([train_samples, features])
                train_labels  = np.concatenate([train_labels, labels])
            else:
                train_samples = features
                train_labels = labels

    # Initialize model
    feature_width = len(train_samples[0])
    total_speakers = len(speakers) # Gives us the total categories
    base_model = get_model('l2', feature_width, 3, 20, .01, total_speakers)

    # train the model against the first 5 utterances of each speaker
    train(base_model, train_samples, train_labels, epochs_n=3)

    # # Load in all features related to a single speaker
    confusion_matrix = test(base_model, king, testing_speakers[0], adv_ms, len_ms, encorder)
    confusion_matrix.plot()
    plt.show()

def format_one_hot(labels):
    categories = []
    for speaker in labels:
        categories.append([speaker])
    return categories

def create_one_hot(speaker_list):
    # Create the list of categories in the
    # format expected of scikit-learn 
    categories = format_one_hot(speaker_list)

    # Generate the one-hot encorder
    one_hot = OneHotEncoder()
    one_hot.fit(categories)
    return one_hot


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
