import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import pickle
from . import similarity

class NaiveEstimator(BaseEstimator): 
    """
    Estimator that applies an averaging heuristic to record feature distributions 
    and match them to make predictions. 
    """
    
    def __init__(self):
        """
        Set up an instance of our naive conflict event detector
        """

    def fit(self, dataset, train_ix, val_ixs): 
        """
        Train the naive model, which tries to characterizre the distribution of features it sees across every 
        cell furnished and all time steps (so we'll end up with story_width x story_width baselines) for each 
        class. During prediction we'll try to match the distribution and predict the class with the best match. 
        """
        # Preallocate our feature and label sets
        features = np.zeros((dataset.story_width, dataset.story_width, dataset.feature_ixs['count'] + 1), dtype=np.float32)    
        labels = np.zeros((dataset.story_width, dataset.story_width))    
        
        # Train the naive model on all stories
        for ix in train_ix: 

            story = dataset.get_story(ix)
            ds = dataset.densify_story(story)
            dl = dataset.label_story(story)
            
            # Gather the mean feature and label structure for each cell
            features = features + np.mean(ds, axis=2)

            # TODO: we are presuming anything that's not a positive label is a negative label and we can't really do that, we should only 
            # be averaging positive labels with negative labels, right? i.e. no reported information doesn't confirm a conflict event isn't 
            # present
            labels = labels + dl
            
        self.model = {}
        
        # Normalize
        self.model['features'] = np.divide(features, len(dataset.stories))
        self.model['labels'] = np.divide(labels, len(dataset.stories))

        return self
        
    def predict(self, dataset, ixs) -> np.ndarray: 
        """
        Predict events based on the crude map we put down during training. 
        """
    
        tqdm.write(f"Generating predictions... ")

        preds = np.zeros((len(ixs), dataset.story_width, dataset.story_width), dtype=np.float32)
        for i, ix in tqdm(enumerate(ixs), total=len(ixs)): 
            story = dataset.get_story(ix)
            ds = dataset.densify_story(story)
            
            # Collapse the temporal dimension, leaving only a scalar (the mean) for each feature
            features = np.mean(ds, axis=2)       

            # Restrict our search for matching distributions to non-zero featuresets
            sample_ixs = np.nonzero(ds)
            model_ixs = np.nonzero(self.model['features'])
            scores = np.zeros((dataset.story_width, dataset.story_width), dtype=np.float32)

            # For every non-zero feature, find the optimal match from our training distributions
            for sx, sy in zip(sample_ixs[0], sample_ixs[1]): 
                for mx, my in zip(model_ixs[0], model_ixs[1]):                     

                    score = similarity.pearson_similarity(features[sx, sy], self.model['features'][mx, my])
                    if score > scores[sx, sy]: 
                        
                        # TODO: validate this again before submission
                        #print(f"Found better match ({score}) for cell {sx}, {sy}")
                        #print(f"Compared {features[sx, sy]} with {self.model['features'][mx,my]}")

                        scores[sx, sy] == score
                        preds[i, sx, sy] = self.model['labels'][mx, my]

        return preds
    
    def score(self, dataset, preds, test_ixs):
        """
        Score a set of predictions
        """       
       
        return similarity.score(dataset, preds, test_ixs)        

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

def train(dataset, train, val): 
    """
    'Train' the naive model 
    """
    return NaiveEstimator().fit(dataset, train, val)

def test( dataset, model, test):
    """
    Test the naive model 
    """
    preds = model.predict(dataset, test)
    scores = model.score(dataset, preds, test)
    print(similarity.score_ixs)
    print(scores) 