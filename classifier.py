

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
    model.compile()
    model.fit(x=features, 
              y=labels, 
              validation_split=.1, # data is selected from the last samples in the x and y data provided
              shuffly=True,
              epochs=epochs_n)
    
    # Run metrics?
    #TODO: View Accuracy
    print(f"Accurance of Model")

    return model

def split_feature_label(speakers, corpus, encoder, adv_ms, len_ms):
    samples = []
    labels  = []
    for i in speakers:
        for file in corpus[i]:
            features, new_labels = get_features(file, adv_ms, len_ms, i)
            new_labels = encoder.transform(new_labels.reshape(-1, 1)).A

            if(samples is not None):
                samples.append(features)
                labels.append(new_labels)

    samples = np.concatenate(samples)
    labels  = np.concatenate(labels)

    return samples, labels


def test(model, corpus, test_utterances, adv_ms, len_ms, one_hot_encoder):
    """
    Execute all, print error rate and return confusion matrix
    """ 
    x, y = split_feature_label(test_utterances, corpus, one_hot_encoder, adv_ms, len_ms):
    one_hot_encoded_label = one_hot_encoder(y[0])
    target_label = test_utterances

    # Returns an array with containing 1 entry for each utterance given
    # Each element in the array is the speaker the model thought the utterance belonged to
    predictions = model.predict(x) 

    # Assume that frames are independent and take the product of the probabilities. 
    # Do so in the log domain (multiplication becomes addition) or your computation will underflow.
   
    # Summation across columns where each column holds the
    # probability of a speaker's identification  
    probability_per_feature = np.sum(predictions, axis=0)
    
    # argmax retuns the index with the highest prediction score
    index_predicted = np.argmax(probability_per_feature)
    # Find the index that occurs the most
    target_predicted = corpus.category_to_speaker(index_predicted)
    
    print(f"Actual Speaker: {target_label}")
    print(f"Predicted Speaker: {target_predicted}")

    # Figure out how many of those labels were the target speaker
    error_rate = correct_predictions / len(y)

    # Add prediction to the confusion matrix
    n=1
    np.zeros((n,n))

    # Confusion Matrix for each speaker classified
    conf_matrix = ConfusionMatrixDisplay(display_labels=test_utterances)

    disp = ConfusionMatrixDisplay(confusion_matrix=[predictions, labels],
                                display_labels=test_utterances[0])

    print(f"Error Rate: {error_rate}")

    return conf_matrix


