
# Add-on packages
import numpy as np

# Short-time Fourier transform
from librosa import stft  # If there is an issue with this pip uninstall and re-install soundfile "pip soundfile"
from librosa.feature import melspectrogram

# Project packages
from library.conversions import ms_to_samples
from library.audio_io import read_wav
from library.endpointer import speech_detector
from library.visualize import plot_spectrogram

def format_training_features(utterences, corpus, encoder, adv_ms, len_ms):
    """
    description
        Formats features and labels for training.
        Utterances must be given in increasing categorical order. 
        i.e 
            index 0 in utterenace corresponding to all audio files 
            that corresponding to the first category in the corpus
    parameters
        utterences - the audio files containing utterances features will be extracted from
        corpus     - the corpus used to maintain categories and speakers
        encoder    - strategy for how labels should be encoded
        adv_ms     - frame advance in milliseconds
        len_ms     - Length of frames in milliseconds
    returns
        samples - all samples
        labels - in corresponding order to the samples, encoded
    """
    samples = []
    labels  = []
    current_category = 0 
    for speaker_utterences in utterences:
        speaker_id = corpus.category_to_speaker(current_category)

        for file in speaker_utterences: 
            # Retrieve features and labels from audio file
            features, new_labels = get_features(file, adv_ms, len_ms, speaker_id, debug=False)
            # Format labels to be accepted by the encoder and produce encoded labels
            new_labels = encoder.transform(new_labels.reshape(-1, 1)).A

            samples.append(features)
            labels.append(new_labels)

        # Move onto the next category
        current_category += 1

    # Concatenate to match expected format for training
    # Concatenate at end to save on memory allocation operation
    samples = np.concatenate(samples)
    labels  = np.concatenate(labels)

    return samples, labels

def get_features(filename: str, adv_ms:float, len_ms: float, label: int,
                 spectral_means_subtraction=True,
                 feature="dB",
                 debug=False):
    """
    
    :param filename: 
    :param adv_ms: 
    :param len_ms: 
    :param label: speaker number
    :param spectral_means_subtraction: noise reduction
    :param feature:  Feature to be computed:  
       dB - spectrogram log mag^2 power in dB
       mel - perceptually filtered spectrogram using a Mel filterbank
           Number of mel filters is based librosa defaults
    :param debug: 
    :return: 
    """

    data = read_wav(filename)  # Read data into memory

    # Find where the speech is
    speechI = speech_detector(data.samples, data.Fs, adv_ms, len_ms)

    # Frame the speech and extract Mel-filtered features
    adv_N = ms_to_samples(data.Fs, adv_ms)
    len_N = ms_to_samples(data.Fs, len_ms)
    # Convert samples to floating point (librosa requirement) and
    # compute discrete Fourier transfrm of each frame
    # Default uses Hann window
    pcm = data.samples.astype(float).T
    # Returns complex DFT
    eps = 1e-6  # we will add a small epsilon to prevent log 0
    specgram = stft(y=pcm, hop_length=adv_N,
                    win_length=len_N, n_fft=len_N)
    # Magnitude spectrogram
    mag_specgram = np.abs(specgram)
    
    if feature == "dB":
        # Convert complex DFT to dB: 10 log10 mag^2,
        specgram = 20 * np.log10(mag_specgram + eps)
        features = specgram
    elif feature == "mel":
        # Computes Mel power (squared)
        filters = 18 if data.Fs <= 8000 else 24
        melgram = melspectrogram(S=mag_specgram, sr=data.Fs,
                                 n_mels=filters, n_fft=len_N)
        # Convert to dB
        melgram = 10 * np.log10(melgram + eps)
        features = melgram
    else:
        raise ValueError(f"Bad feature specification {feature}")

    features = np.squeeze(features)  # Remove singleton
    # Spectrogram is now frequency X frames

    noiseI = np.logical_not(speechI)  # indicator for noise

    # Use spectral means subtraction *Noise Reduction technique
    if spectral_means_subtraction:
        # Find mean of each frequency bin across the noise regions and
        # normalize spectrogram
        mean = np.mean(features[:,noiseI], axis=1)
        # Can only subtract a vector if it matches the last axis,
        # so we transpose to frames X frequency, subtract the mean
        # and put it back
        features = (features.T - mean).T

    if debug:
        # Need to fix for Mel frequencies, displays everything as linear
        plot_spectrogram(features, adv_ms, len_ms, speechI)

    # Delta features approximate derivatives of features.
    # This is not used in this set of experiments.
    delta = 0
    if delta:
        # Only implemented for delta == 1, will probably break with longer deltas
        # Compute the diff +/- delta steps
        padded = np.pad(features, ((0,), (delta,)), mode='edge')
        slope = padded[:,delta+1:] - padded[:,0:padded.shape[1]-(delta+1)]
        features = np.concatenate((features, slope), axis=0)

    # Drop portions of spectrogram that are associated with noise
    speech_features = np.delete(features, noiseI, axis=1)

    # Create label array - must have a speaker label for each
    # frame of speech associated with the speaker
    labels = np.ones(speech_features.shape[1]) * label

    # Tensorflow will expect examples X features (frequencies)
    return speech_features.T, labels






