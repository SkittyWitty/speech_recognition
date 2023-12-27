from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib

# project imports
from library.corpus import King
from feature_extraction import format_training_features
from architecture import get_model
from classifier import train, test
from sklearn.preprocessing import OneHotEncoder

# Post Assigmnet turn in notes
# Load Test and Train seperately not all at the same time
# There are comments on the the classifier that I need to review

def main(args):
    # Specify how matplotlib renders (backend library)
    matplotlib.use("QtAgg")
    plt.ion()

    # Parsing framing parameter
    adv_ms = args.adv_ms
    len_ms = args.len_ms

    # Setting up the corpus to be used
    king = King(args.king, group="Nutley")
    speakers = list(king.get_speakers())
    total_speakers = len(speakers) # Gives us the total categories

    # Create an encoder that coverts a category to a one-hot enocoding
    encorder = create_one_hot_encoder(speakers)

    # Split utterances into test and training sets
    train_test_split = 5
    training_utterences = []
    testing_utterences = []
    for speaker in speakers:
        # Each index corresponds to a speaker category hence we must maintain the ordering
        speaker_files = king[speaker]

        # Split utterences in each category to test and training
        # based on split parameter
        training_utterences.append(speaker_files[:train_test_split])
        testing_utterences.append(speaker_files[train_test_split:])

    # Prepare training utterances
    train_samples, train_labels = format_training_features(training_utterences, king, encorder, adv_ms, len_ms)

    # Initialize model
    feature_width = len(train_samples[0])
    model         = get_model('l2', feature_width, 2, 50, .01, total_speakers)
    model.summary() # Print specifications of model created

    # Train the model against the first 5 utterances of each speaker
    train(model, train_samples, train_labels, epochs_n=5)

    # Test on the remaining utterances
    confusion_matrix = test(model, king, testing_utterences, adv_ms, len_ms)

    # Display the Confusion Matrix
    confusion_matrix.plot()
    plt.show()

    print("Artificial Breakpoint Stop")

def create_one_hot_encoder(speaker_list):
    """
    description
        Creates a one hot encoder based on the category's provided
    parameters
        speaker_list - all categories that may be predicted in the model
    returns
        One-hot Encoder
    """
    # Format the list of categories
    categories = []
    for speaker in speaker_list:
        categories.append([speaker])

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
