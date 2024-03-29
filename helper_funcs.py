import numpy as np
import pandas as pd
import Species_description
import keras.backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1_score(y_true, y_pred):
    """
    Computes the F1 score of the predictions.
    """
    # Convert predictions to binary values (0 or 1)
    y_pred = K.round(y_pred)

    # Calculate true positives, false positives, and false negatives
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())

    # Return F1 score as a metric
    return K.mean(f1_score)

def feature_extract(specie, feature_name):
    if specie in Species_description.description.keys():
        specie_features = Species_description.description[specie]
        if feature_name in Species_description.description[specie].keys():
            specie_feature = specie_features[feature_name]
            output = specie_feature
        else: # for the null input case
            print("need input")
            output = ''
    else: # for species not found
        output = ''
    return output
