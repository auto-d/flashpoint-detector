import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle
from . import similarity
from sklearn.metrics import f1_score, precision_recall_curve

class NaiveEstimator(BaseEstimator): 
    """
    Estimator that applies a heuristic to predict future events 
    """
    
    def __init__(self):
        """
        Set up an instance of our naive estimator 
        """

    def fit(self, train, val, val_chk): 
        """
        Fit our naive estimator
        """ 

        return self
        
    def predict(self, df) -> np.ndarray: 
        """
        Predict events
        """
        events = []
                
        tqdm.write(f"Generating predictions... ")
    
    def score(self, preds):
        """
        Score a set of predictions
        """        

        tqdm.write(f"Scoring predictions... ")

        return map

def save_model(model, path):
    """
    Save the model to a file
    NOTE: copy/pasta from nlp project 
    """    
    filename = os.path.join(path, 'naive.pkl')

    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    tqdm.write(f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    NOTE: copy/pasta from nlp project 
    """
    model = None

    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f) 
    
    if type(model) != NaiveEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(train, val): 
    """
    'Train' the naive model 
    """
    return NaiveEstimator().fit(train, val)

def test(model, test, preds):
    """
    Test the naive model 
    """
    preds = model.predict(test)
    model.score(preds, test)