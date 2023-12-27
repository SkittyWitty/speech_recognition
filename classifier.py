import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from feature_extraction import get_features

# Development debug messages
debug = False

def train(model, features, labels, epochs_n=20):
    """
    description
        Trains and performs validation on the given model. 

        Reference Kera Model Training APIs
        https://keras.io/api/models/model_training_apis/
    parameters
        model    - architected keras model
        features - training features
        labels   - training labels
        epochs   - how many epochs to be used in training
    returns
        model - ready to be use perform predictions~!
    """
    # Split training data with a 90/10 random split 
    # Must be split in this way as keras will not shuffle the data via the validation parameter
    X_train, X_validation, y_train, y_validation= train_test_split(features, labels, test_size=0.1)
    
    # Adam: Current most popular optimizer
    # Catagorical Crossentropy: Used as we are categorizing spakers
    # Metric: How accurate is was our training at each epoch?
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ["accuracy"])
    # Performs the training
    model.fit(x=X_train, 
              y=y_train, 
              shuffle=True,
              validation_data=(X_validation, y_validation),
              epochs=epochs_n,
              batch_size=100) # recommended was 100
    
    return model


def test(model, corpus, test_utterances, adv_ms, len_ms):
    """
    description
        Test the model and calculate error rate and
        confusion matrix based on the predictions made. 
    parameters
        model           - the model that will be performing the predictions
        corpus          - corpus feature were dervied from
        test_utterances - what utterances will be used to perform predictions 
        adv_ms          - frame advance in milliseconds
        len_ms          - Length of frames in milliseconds
    returns
        cm - confusion matrix that is ready to be displayed/plotted
    """ 
    # Keep track of predictions for each utterence
    n_utterences = len(test_utterances)
    speaker_predictions = np.zeros((n_utterences, n_utterences))
    
    # Test utterance's are numerically ordered
    # Begin at 0 and increment when done predicting utterences for a speaker
    target_category = 0

    # Small epsilon used to pad probabilities.
    # So no prediction probability will be 0. 
    eps = 1e-6

    # Parse through all utterences for each speaker
    for speaker_utterences in test_utterances:

        speaker = corpus.category_to_speaker(target_category)

        # For each audio file (utterence) extract features and perform a prediction
        for audio_file in speaker_utterences: 
            features, _ = get_features(audio_file, adv_ms, len_ms, speaker, debug=False)

            p = model.predict(features)
            # Assume that frames are independent and take the product of the probabilities
            # Multiplication becomes addition due to log domain
            p_fused = np.sum(np.log(p+eps), axis=0)

            # return the category with the highest prediction score
            predicted = np.argmax(p_fused)

            # Keep track of our results
            speaker_predictions[target_category][predicted]+= 1

            # Log the predictions as they are made
            if (debug):
                speaker_predicted = corpus.category_to_speaker(predicted)
                print(f"Actual Speaker: {speaker}")
                print(f"Predicted Speaker: {speaker_predicted}")

        # Prepare to predict on utterences of the next speaker
        target_category += 1

    # Create a confusion matrix using our track record
    cm = confusion_matrix_calc(speaker_predictions, corpus.get_speakers())
    return cm


def confusion_matrix_calc(speaker_predictions, speaker_list):
    """
    description
        Creates confusion matrix and calculates error rate
    parameters
        speaker_predictions - The predictions that were made
        speaker_list        - The speakers we predicted for
    returns
        cm_disp - confusion matrix that is ready to be displayed/plotted
        + prints the error rate
    """
    correct_predictions = np.trace(speaker_predictions) #might have had to use np.diag
    total_predictions = np.sum(speaker_predictions)

    # Figure out how many of those labels were the target speaker
    error_rate =  1.0 - (correct_predictions / total_predictions)

    # Confusion Matrix for each speaker classified
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=speaker_predictions,
                                display_labels=speaker_list)

    print(f"Error Rate: {error_rate}")
    return cm_disp