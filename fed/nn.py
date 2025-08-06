import os
import torch 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from . import similarity
from .dataset import FlashpointsTorchDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from . import naive 

class FlashpointsCNN(nn.Module):
    """
    Convolutional neural network to infer and act on relationships in the flashpoints
    event database. 
    
    Base loosely CNN network geometry described by Huang et al: 
    [4] Huang JP., Wang XA., Zhao Y., Xin C., Xiang H. (China) 
    Large earthquake magnitude prediction in Taiwan based on deep learning neural network, 149-160
    https://nnw.cz/doi/2018/NNW.2018.28.009.pdf
    """

    def __init__(self, width=20, depth=7, features=7):
        """
        Construct our network 

        N/batch : batch dimension 
        C/channels : we will assign a channel to each feature in our story
        D/depth : this will be our temporal dimension
        H/W : this is our square spatial dimension
        """
        self.input_width = width
        self.input_depth = depth 

        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=features, out_channels=3, kernel_size=(5,5,5)) 
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=(2,2,1))
        self.conv2 = nn.Conv3d(in_channels=features, out_channels=3, kernel_size=(3,3,2)) 
        self.fc1 = nn.Linear(6*6*1*7, width*width) 
        
    def forward(self, x): 
        """
        Forward pass through the network. Make sure zero inputs (non-detections) are 
        masked during backpropagation or the zeros will dominate the gradient
        """

        # (b, 20, 20, 7, 7) -> apply 5x5x5 kernel, 7 channels 
        x = self.conv1(x)
        x = F.relu(x) 

        # (b, 16, 16, 3, 7) -> apply 2x2x2 @ stride (2,2,1)
        x = self.pool1(x)

        # (b, 8,8,2,7) -> apply 3x3x2 kernel
        x = self.conv2(x) 
        x = F.relu(x) 

        # (b, 6,6,1,7) -> 400 
        x = self.fc1(x.flatten()) 
        x = F.relu(x) 
        
        #TODO: remove and test, loss function wants unnormalized logits
        #x = F.sigmoid() 

        return x
    
class FlashpointsEstimator(): 
    """
    Overarching abstraction for our nn-based flashpoint event predictor
    """

    def __init__(self, tensorboard_dir="./runs"): 
        """
        Initialize an object 
        """
        self.module = None
        self.tensorboard_dir = tensorboard_dir

    def train(self, dataset, val_ixs, epochs=1, lr=0.001, loss_interval=10):
        """
        Train the model with the furnished stories (spatio-temporal bricks that 
        capture a sequence of days culminating in one or more labeled events in the 
        associated aperture)

        NOTE: training loop based on prior submissions for NN training in torch
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_loss = []

        model = FlashpointsCNN()  
        loader = dataset.get_data_loader()            
        
        # TODO: these need to be normalized 
        val_stories = []
        for ix in val_ixs: 
            story = dataset.get_story(ix)
            val_stories.append = dataset.densify_story(story)

        # Track progress with tensorboard-style output
        tqdm.write(f"Logging tensorboard output to {self.tensorboard_dir}")
        writer = SummaryWriter(os.path.join(self.tensorboard_dir), 'fp_classifier')

        # Rehome, if necessary 
        model = model.to(device)
        
        # We are looking to produce a probability that an event occurred at each 
        # cell in the story grid, 
        loss_fn = nn.BCELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        tqdm.write(f"Starting training run...")
        for epoch in tqdm(range(epochs), total=epochs):
        
            running_loss = 0.0
            for i, story in tqdm(enumerate(loader), total=len(dataset)/dataset.batch_size):

                # Mask out 0 inputs, we can't presume a lack of a thermal detection 
                # suggest a lack of corresponding activity given overflight, atmospheric
                # attenuation, cloud cover, etc... this is essential as well  
                # given the sparsity of our detections. The model could learn to just emit 
                # zeros and score reasonably well otherwise. 
                mask = (story > 0)
                mask = mask.to(device)
                story = story.to(device)
                
                # Toss any gradient residue from prior runs
                optimizer.zero_grad()

                # Forward + back, masking out absent data
                outputs = model(story)
                loss = loss_fn(outputs[mask], story[mask])
                loss.backward()

                optimizer.step()

                # Accumulate metrics for hyperparameter tuning
                running_loss += loss.item()

                writer.add_scalar(f"Training loss", loss, epoch * len(loader) / loader.batch_size)

                # TODO: log validation data based on our validation indices (stories)
                val_loss = None
                for val_story in val_stories: 
                    preds = model(val_story)
                    # ! 

                writer.add_scalar(f"Validation loss", val_loss, epoch * len(loader) / loader.batch_size) 
                
                if (i % loss_interval) == (loss_interval - 1): 
                    interval_loss = running_loss / loss_interval                    
                    tqdm.write(f"[{epoch + 1}, {i + 1:5d}] loss: {interval_loss:.5f}")
                    running_loss = 0 
        
        # Update our object state
        self.model = model 

        writer.close()
        tqdm.write("Training complete!") 

        return model, train_loss 

    def predict(self, dataset) -> np.ndarray: 
        """
        Predict conflict events given a torch dataset
        """
        preds = []
            
        # Avoid gradient bookkeeping
        with torch.no_grad(): 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self.model.to(device) 
            
            # Avoid training interventions like batch norm and dropout
            model.eval() 
            
            # Generate recommendations
            loader = dataset.get_data_loader()

            tqdm.write(f"Generating predictions... ")

            preds = np.zeros((len(loader), dataset.dataset.story_width, dataset.dataset.story_width))
            for i, story in tqdm(enumerate(loader)): 
                
                story.to(device)
                preds[i] = model(story)

        return preds

    def score(self, dataset, preds, test_ixs):
        """
        Score our predictions
        """        

        return naive.score(dataset, preds, test_ixs)

def save_model(model, path):
    """
    Save the model to a file
    """
    filename = os.path.join(path, "nn.pt")
    torch.save(model, filename)
    print(f"Model saved to {filename}")

    return filename

def load_model(path): 
    """
    Pull a saved model off disk 
    """
    model = torch.load(os.path.join(path, "nn.pt"), weights_only=False)
    
    if type(model) != FlashpointsEstimator: 
        raise ValueError(f"Found unexpected type {type(model)} in {path}!")

    return model

def train(train, val, epochs):
    """
    Train the autoencoder given the provided dataset     
    """
    model = FlashpointsEstimator()
    model.train(train, val, epochs=epochs)
    return model 

def test(model, test):
    """
    Test the model
    """
    preds = model.predict(test)
    model.score(test, preds)
