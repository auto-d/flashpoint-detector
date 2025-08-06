
from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from . import similarity
from . import naive

class ClassicEstimator(BaseEstimator):
    """
    A random forest estimator that uses a reduced subset of the available data to 
    predict whether or not conflict events occurred in a region that slides around
    the Ukraine data
    """ 
    
    def __init__(self):
        """
        Initialize a new instance of the model 
        """
        self.model = None
    
    def fit(self, dataset, train, val): 
        """
        Fit our estimator given a training and validation set along with our 
        core data abstraction class which probides help along the way for transformations
        and lookups. 
        """ 

        tqdm.write(f"Flattening {len(train)} stories in preparation for training...")
        X = dataset.flatten_stories(train)
        y = dataset.flatten_labels(train)
        
        tqdm.write(f"Flattening complete! New training shape is {X.shape}, new label shape is {y.shape}.")

        tqdm.write(f"Fitting random forest...")
        self.model = RandomForestClassifier(max_depth=10, random_state=42)
        self.model.fit(X, y)

        tqdm.write("Training complete!")

        return self

    def predict(self, dataset, ixs) -> np.ndarray: 
        """
        Generate classifications for the provided stories
        """
        tqdm.write(f"Running predictions... ")
        
        X = dataset.flatten_stories(ixs)
        preds = self.model.predict(X)

        self.score(dataset, preds, ixs)

        return preds
    
    def score(self, dataset, preds, test_ixs):
        """
        Score our predictions - same logic applied in our naive model
        """        
        return similarity.score(dataset, preds, test_ixs)

def save_model(model, path):
    """
    Save the model to a file
    NOTE: copy/pasta from recommenderes project
    """    
    filename = os.path.join(path, 'classic.pkl')

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    tqdm.write(f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    NOTE: copy/pasta from vision project 
    """
    model = None

    filename = os.path.join(path, 'classic.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f)         
    
    if type(model) != ClassicEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(dataset, train, val): 
    """
    Train the model     
    """
    model = ClassicEstimator() 
    model.fit(dataset, train, val)
    return model     

def test(dataset, model, test):
    """
    Test the RF model 
    """
    preds = model.predict(dataset, test)
    model.score(dataset, preds, test)
