

# add-on modules

from keras.callbacks import TensorBoard
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from feature_extraction import get_features

def train(model, features, labels, epochs_n=20):
    """
    Pass in network model, training feature matrix and labels
    train for specified number of epochs_n (use a low number
    while developing your code

    Reference Kera Model Training APIs
    https://keras.io/api/models/model_training_apis/
    """
    # Split your training data (see scipy) with a 90/10 split to use 90% of it for 
    # training and 10% for validation. The validation data can be provided to keras’s fit 
    # function and can show you how your model is performing during training. Do not expect 
    # wonderful accuracy at this stage.
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.fit(x=features, 
              y=labels, 
              shuffle=True,
              epochs=epochs_n,
              batch_size=20)
    
    return model

def get_x_y(speaker, corpus, encoder, adv_ms, len_ms):
    """
    
    """
    samples = []
    labels  = []

    for file in corpus[speaker]:
        features, new_labels = get_features(file, adv_ms, len_ms, speaker)
        new_labels = encoder.transform(new_labels.reshape(-1, 1)).A

        samples.append(features)
        labels.append(new_labels)

    samples = np.concatenate(samples)
    labels  = np.concatenate(labels)

    return samples, labels


def test(model, corpus, test_utterances, adv_ms, len_ms, one_hot_encoder):
    """
    Prints the error rate
    return
        confusion matrix - ready for display via matplotlib
    """ 
    # Keep track of predictions
    speaker_predictions = np.zeros((25, 25))

    # Collects features and labels
    for speaker in test_utterances:
        x, y = get_x_y(speaker, corpus, one_hot_encoder, adv_ms, len_ms)
        target_label = corpus.speaker_category(speaker)

        # Returns an array with containing 1 entry for each utterance given
        # Each element in the array is the speaker the model thought the utterance belonged to
        frame_predictions = model.predict(x) 

        # Assume that frames are independent and take the product of the probabilities
        # Place predictions into log domain
        frame_predictions = np.log(frame_predictions)

        # Multiplication becomes addition due to log domain
        # Summation across columns where each column holds the probability of a speaker's identification  
        probability_per_category = np.sum(frame_predictions, axis=0)
        
        # argmax retuns the index with the highest prediction score
        index_predicted = np.argmax(probability_per_category)

        # Update the tracker
        speaker_predictions[target_label][index_predicted]+= 1
        
        # Find the index that occurs the most
        speaker_predicted = corpus.category_to_speaker(index_predicted)
        print(f"Actual Speaker: {speaker}")
        print(f"Predicted Speaker: {speaker_predicted}")

    # Create a confusion matrix using our track record
    conf_matrix = confusion_matrix_calc(speaker_predictions, corpus.get_speakers())
    return conf_matrix


def confusion_matrix_calc(speaker_predictions, speaker_list):
    """
    speaker_predictions - The predictions that were made
    speaker_list - The speakers we predicted for
    """
    correct_predictions = np.trace(speaker_predictions)
    total_predictions = np.sum(speaker_predictions)

    # Figure out how many of those labels were the target speaker
    error_rate =  1.0 - (correct_predictions / np.sum(total_predictions))

    # Confusion Matrix for each speaker classified
    disp = ConfusionMatrixDisplay(confusion_matrix=speaker_predictions,
                                display_labels=speaker_list)

    print(f"Error Rate: {error_rate}")
    return disp