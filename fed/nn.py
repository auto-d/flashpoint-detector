import os
import math 
from datetime import datetime
import torch 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import similarity
from dataset import FlashpointsTorchDataset

class Autoencoder(nn.Module):
    """
    Autoencoder

    NOTE: with cues from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
    """

    def __init__(self, dims, l1, l2):
        """
        Initialize a new object given an item count 
        """
        self.dims = dims
        self.l1 = l1
        self.l2 = l2 

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dims, l1),
            nn.ReLU(), 
            nn.Linear(l1, l2),
            nn.ReLU(), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(l2, l1),
            nn.ReLU(), 
            nn.Linear(l1, dims),
            nn.ReLU(), 
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Implement our forward pass 
        """
        h = self.encoder(x) 
        r = self.decoder(h)

        return r

class AutoencoderEstimator(): 

    def __init__(self, tensorboard_dir="./runs", l1 = 250, l2 = 50): 
        """
        Initialize an object 
        """
        self.module = None
        self.tensorboard_dir = tensorboard_dir
        self.l1 = l1
        self.l2 = l2 

    def train(self, dataset, val, val_chk, epochs=2, lr=0.0005, loss_interval=10):
        """
        Train the model with the provided user-item dataset, optionally furnishing a learning 
        rate and interval to plot loss values
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_loss = []

        u_map, i_map = dataset.get_mappings()
        model = Autoencoder(dims=len(i_map), l1=self.l1, l2=self.l2)  
        loader = dataset.get_data_loader()
            
        # Track progress with tensorboard-style output
        tqdm.write(f"Logging tensorboard output to {self.tensorboard_dir}")
        writer = SummaryWriter(os.path.join(self.tensorboard_dir), 'exp_ae_recommender')

        # Rehome, if necessary 
        model = model.to(device)
        
        # We'll use MSE since we're interested in correctly reproducing ground-truth reviews
        # the value of the autoencoder will be in the values it infers where no rating was 
        # provided (which will form our recommendations)
        loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        tqdm.write(f"Starting training run...")
        for epoch in tqdm(range(epochs), total=epochs):
        
            running_loss = 0.0
            for i, reviews in tqdm(enumerate(loader), total=len(dataset)/dataset.batch_size):

                # Build a mask to apply later 
                mask = (reviews > 0)
                
                # Push our key matrices to whatever device we've got                 
                mask = mask.to(device)
                reviews = reviews.to(device)
                
                # Toss any gradient residue from prior runs
                optimizer.zero_grad()

                # Run the reviews through the network and then propagate the gradient
                # backward to improve our alignment with ground-truth review. 
                # Note we mask out any non-reviews to avoid the network learning 
                # to reconstruct, as our prediction is based entirely on the network's 
                # ability to estimate these so we want them to evolve with the other
                # weights (and we wouldn't know which way to push them anyway)
                outputs = model(reviews)
                loss = loss_fn(outputs[mask], reviews[mask])
                loss.backward()

                optimizer.step()

                # Accumulate metrics for hyperparameter tuning
                running_loss += loss.item()

                writer.add_scalar(f"training loss", loss, epoch*len(loader)/loader.batch_size)

                if (i % loss_interval) == (loss_interval - 1): 
                    interval_loss = running_loss / loss_interval                    
                    tqdm.write(f"[{epoch + 1}, {i + 1:5d}] loss: {interval_loss:.5f}")
                    running_loss = 0 
        
        # Update our object state
        self.model = model 
        self.schema = dataset[0]
        self.schema[:] = 0 
        self.u_map = u_map
        self.i_map = i_map 

        hps = { 
            'train_reviews' : len(dataset),
            'train_items' : len(i_map),
            'train_users' : len(u_map), 
            'epochs':epochs, 
            'lr': lr, 
            'batch_size': loader.batch_size,
            'autoencoder_io_dims': model.dims, 
            'autoencoder_hidden_1': model.l1,
            'autoencoder_hidden_2': model.l2,
        }        
        writer.add_hparams(hps, {}) 
        writer.close()
        tqdm.write("Training complete!") 

        return model, train_loss 

    def prepare_new_dataset(self): 
        """
        When recommending for new users, we have no interaction data, provide a 
        bootstrapping method that creates anew dataset to help the client understand
        our schema as well as fill out clicks (to enable prediction)
        """
        ds = DeepCartTorchDataset(ui=np.array([self.schema]), u_map=self.u_map, i_map=self.i_map, batch_size=1)
        return ds

    def recommend(self, dataset, k, reverse=False, allow_ixs=None) -> np.ndarray: 
        """
        Generate top k predictions given a list of item ratings (one per user)
        """
        recommendations = []

        u_map, i_map = dataset.get_mappings()
        
        # The model baseline is is the itemset we were provided at training time, the if data we 
        # are predicting on differs, we have to map these new items into the old item space 
        mapping = True if i_map.keys() != self.i_map.keys() else False            
            
        # Avoid gradient bookkeeping
        with torch.no_grad(): 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self.model.to(device) 
            
            # Avoid training interventions like batch norm and dropout
            model.eval() 
            
            # Generate recommendations
            for u, reviews in tqdm(enumerate(dataset.get_data_loader()), total=len(dataset)/dataset.batch_size):

                # Note the ratings for this user
                rated = np.nonzero(reviews[0]) 
                rated = rated[0].tolist() if len(rated) != 0 else []

                # Map this user's reviews to our training item space if needed 
                if mapping:                     
                    mapped_reviews = torch.zeros((len(reviews), len(self.schema)))
                    for i in range(len(reviews)): 
                        new_ixs = np.nonzero(reviews[i]).flatten()
                        model_ixs = similarity.map_keys(i_map, new_ixs, self.i_map)
                        mapped_reviews[i][model_ixs] = reviews[i][new_ixs]
                    reviews = mapped_reviews 

                reviews = reviews.to(device)
                logits = model(reviews) 

                # Find the top_k novel recommendations 
                output = logits.to("cpu").flatten().numpy()
                recommended = []
                while len(recommended) < k: 
                    best = None
                    if reverse: 
                        best = similarity.argmin(
                            output, 
                            exclude=rated + recommended,
                            include=allow_ixs) 
                    else:
                        best = similarity.argmax2(
                            output, 
                            exclude=rated + recommended, 
                            include=allow_ixs) 

                    recommended.append(best)
                                        
                    # Record stand-in user, recommended item and inferred rating
                    row = [
                        similarity.find_key(u_map, u), 
                        similarity.find_key(self.i_map, best),
                        output[best]
                        ]
                    recommendations.append(row)

        df = pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'prediction']) 
        return df

    def score(self, top_ks, test_chk, k=10):
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
    """
    filename = os.path.join(path, "autoencoder.pt")
    torch.save(model, filename)
    print(f"Model saved to {filename}")

    return filename

def load_model(path): 
    """
    Pull a saved model off disk 
    """
    model = torch.load(os.path.join(path, "autoencoder.pt"), weights_only=False)
    
    if type(model) != AutoencoderEstimator: 
        raise ValueError(f"Found unexpected type {type(model)} in {path}!")

    return model

def train(train, epochs, val, val_chk):
    """
    Train the autoencoder given the provided dataset     
    """
    model = AutoencoderEstimator()
    model.train(dataset=train, val=val, val_chk=val_chk, epochs=epochs)
    return model 

def test(model, test, test_chk, top_k):
    """
    Test the autoencoder model 
    """
    allow_items = list(test_chk.df.item_id.unique())
    allow_ixs = [model.i_map.get(k) for k in allow_items]    
    test_k = min(len(allow_items), top_k)

    top_ks = model.recommend(test, test_k, allow_ixs=allow_ixs)
    model.score(top_ks, test_chk, test_k)
