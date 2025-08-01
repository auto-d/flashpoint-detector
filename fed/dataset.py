import sys 
import os
import pandas as pd
import numpy as np 
import fiona
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
import seaborn as sns
import shapely
import datetime
import kagglehub
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch 

class FlashpointsDataset(): 
    """
    Abstraction for the Flashpoints Ukraine Dataset (FUD) 
    """
    
    def __init__(self, tag): 
        """
        Set up the instance, deferring creation/load given the memory-intensive nature
        """
        self.tag = tag
        self.path = None

    def get_memory_usage(self): 
        """
        Get memory usage for a dataframe in bytes
        """
        return self.df.memory_usage(index=True).sum()

    def download(self, force=False):
        """
        Retrieve the latest version of the dataset from its home on kaggle, 
        optionally forcing the download regardless of whether we have a cached copy. 
        """
        tqdm.write(f"Retrieving FUD from Kaggle (force={force})...")
        self.path = kagglehub.dataset_download("justanotherjason/flashpoint-ukraine-dataset", path="fed.gpkg", force_download=force)

        print(f"Done. Path to local dataset files: {self.path}")

    def load(self):
        """
        Load the dataset into memory 
        """
        self.download()

        if self.path.endswith('.shp'):
            tqdm.write("Found shapefile dataset, loading...")
            gdf = gpd.read_file(self.path)
        elif self.path.endswith('.gdb'):
            gdf = tqdm.write("Found geodatabase, loading...")
            gdf = gpd.read_file(self.path)
        elif self.path.endswith('.gpkg'):
            gdf = tqdm.write("Loading geopackage....")
            gdf = gpd.read_file(self.path)
        else: 
            raise ValueError('Unknown dataset format, {self.path}!')

        print(f"Loaded GUD, with coordinate system {gdf.crs}")  

    def store(self, dir_):
        """ 
        Write our dataset to disk 
        """
        os.makedirs(dir_, exist_ok=True)

        raise NotImplementedError
    
        fud_file = os.path.join(dir_,f"fud_{self.tag}.parquet")
        print(f"Writing {len(self.gdf):,} reviews as {fud_file}...")
        self.reviews.to_parquet(fud_file)

        print(f"Wrote '{self.tag}' dataset to {dir_}.")

    def build(self): 
        """
        Build our complete dataset
        """
        print(f"Building dataset")

        # TODO: any necessary revision of the FUD to support prediction, e.g. incorporation of 
        # terrain/topo 

        print(f"Generation complete!")

    def split(self):
        """
        Splitting data for training and evaluation in the context of a self-exciting process as 
        we are dealing with in conflict events is problematic as: 
        1. prior conflict events influence future conflict events, doing a naive temporal or class-based split 
        both deprives us at prediction time of relevant priors and leaks ground-truth at training time 
        2. ignores the fact that latent patterns we're trying to predict may evolve over time, and a 
        temporal split would potentially remove a large swath of information a model might need to learn. 

        Drawing inspiration from weather prediction models (which have the same problem) we opt for a 
        sliding window


        See conversation with gpt-4o on meteorological approaches: https://chatgpt.com/share/688a362f-7ba4-8013-a4b8-3be2073f8569 

        Generate splits as follows: 
        - self.train : training matrix
        - self.val : matrix to predict on during validation 
        - self.test : matrix for test predictions
        """
        tqdm.write(f"Splitting dataset ({len(self.gdf)} rows)")
        
        #TODO: we have 67K examples of firms detections with an event and 67K without, ensure we are balancing these
        # two classes here, i.e. make sure stratification is using the correct field and logic
        train, test = train_test_split(self.gdf, test_size=0.2, random_state=42, shuffle=True, stratify="a_event")
        val, test = train_test_split(test, test_size=0.5, random_state=42, shuffle=True, stratify="a_event")
                
        tqdm.write(f"Generated splits:\n"\
            f" train : {len(train)} users @ {len(train)} detections\n"\
            f" val   : {len(val)} users @ {len(val)} detections\n"\
            f" test  :{len(test)} users @ {len(test)} detections\n"\
            )

class FlashpointsTorchDataset(torch.utils.data.Dataset):
    """
    Torch-compatible dataset to capitalize on the former's abstraction of batching, 
    parallelism and shuffling memory to GPU & back
    """

    def __init__(self, fud:FlashpointsDataset, batch_size=10): 
        """
        Initialize a new instance given flashpoints dataset object
        """
        self.batch_size = batch_size
        
        self.fud = fud

    def __len__(self): 
        """
        Retrieve length of the dataset
        """
        return len(self.fud) 
    
    def __getitem__(self, idx): 
        """
        Retrieve an item at the provided index
        """
        row = self.fud.iloc[idx]
        # TODO: Normalize, scale, etc? 
        return row
    
    def get_data_loader(self, shuffle=True): 
        """
        Retrieve a pytorch-style dataloader that loads data with this instance
        """
        loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)        
        return loader