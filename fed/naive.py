import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle
from ..src import similarity

class NaiveEstimator(BaseEstimator): 
    """
    Estimator that grabs the most popular items out of the user-item matrix 
    and vomits them back at prediction time.
    """
    
    def __init__(self):
        """
        Set up an instance of our naive estimator 
        """
        self.item_ratings = None

    def fit(self, train, val, val_chk): 
        """
        Fit our naive estimator
        """ 
        ui, u_map, i_map = train.gen_affinity_matrix() 

        tqdm.write(f"Fitting model to review data... ")
        # Assemble a list of the mean ratings for reviewed items
        item_ratings = [0] * len(i_map)
        for i in tqdm(i_map.values()): 
            
            ratings = []
            for u in u_map.values(): 
                if ui[u][i] != 0: 
                    ratings.append(min(ui[u][i], 5))
            
            item_ratings[i] = np.mean(ratings)
        
        self.item_ratings = item_ratings
        self.model_u_map = u_map
        self.model_i_map = i_map 
         
        return self
        
    def recommend(self, ui, k, allow_ixs=None) -> np.ndarray: 
        """
        Generate top k predictions given a list of item ratings (one per user)        
        """
        recommendations = []
                
        tqdm.write(f"Running predictions... ")
        
        # Map this (probably smaller and potentially disjoint) item vector into 
        # our training baseline

        ui, u_map, i_map = ui.gen_affinity_matrix() 
        
        # For each requested user, find the best-reviewed items that they have't
        # already reviewed... 
        for u in tqdm(range(len(u_map))): 
            
            # This new dataset is unlikely to share the same item space as our 
            # training set. Map the item indices to corresponding indices in our 
            # item baseline. 
            new_rated = list(np.nonzero(ui[u])[0]) 
            rated = similarity.map_keys(i_map, new_rated, self.model_i_map)
            recommended = []
            
            while len(recommended) < k: 
                best_rated = similarity.argmax(
                    self.item_ratings, 
                    exclude=rated + recommended,
                    include=allow_ixs)
                recommended.append(best_rated) 
                
                # Recommendations need to be in a format suitable for scoring w/ the 
                # Recommenders MAP@K. I.e. dataframe with cols user, item & rating             
                row = [
                    similarity.find_key(u_map, u), 
                    similarity.find_key(self.model_i_map, best_rated), 
                    self.item_ratings[best_rated]
                    ]
                recommendations.append(row)
        
        df = pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'prediction']) 
        return df 
    
    def score(self, top_ks, test_chk, k):
        """
        Employ the recommenders library to calculate MAP@K here. 
        NOTE: Recommenders examples used to source call semantics, see e.g.
        https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb
        """        

        tqdm.write(f"Scoring recommendations... ")

        map = map_at_k(
            test_chk.df, 
            top_ks, 
            col_item="item_id", 
            col_user="user_id", 
            col_prediction='prediction', 
            k=k)
        tqdm.write(f"MAP@K (k={k}): {map}")

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

def train(train, val, val_chk): 
    """
    'Train' the naive model 
    """
    return NaiveEstimator().fit(train, val, val_chk)

def test(model, test, test_chk, top_k):
    """
    Test the naive model 
    """
    
    # We need to constrain comparison here... the matrix is too sparse to expect 
    # any reasonable rankings otherwise.
    allow_items = list(test_chk.df.item_id.unique())
    allow_ixs = [model.model_i_map.get(k) for k in allow_items]        
    test_k = min(len(allow_items), top_k)
    
    top_ks = model.recommend(test, test_k, allow_ixs=allow_ixs)
    model.score(top_ks, test_chk, test_k)