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
from sklearn import train_test_split
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
        self.path = kagglehub.dataset_download("justanotherjason/flashpoint-ukraine-dataset", force_download=force)

        print(f"Done. Path to local dataset files: {self.path}")

    def load(self):
        """
        Load the dataset into memory 
        """
        self.download()

        if self.path.endswith('.shp'):
            tqdm.write("Found shapefile dataset, loading...")
            #TODO: we used the layer arg here, but I think maybe not needed if we only wrote one layer. clarify. 
            gdf = gpd.read_file(self.path)
        elif self.path.endswith('.gdb'):
            gdf = tqdm.write("Found geopacakage dataset, loading...")
        else: 
            raise ValueError('Unknown dataset format, {self.path}!')

        self.crs = gdf.crs  

    def build(self): 
        """
        Build our complete dataset
        """
        print(f"Building dataset")

        # TODO: any necessary revision of the FUD to support prediction, e.g. incorporation of 
        # terrain/topo 

        print(f"Generation complete!")

    def store(self, dir_):
        """ 
        Write our dataset to disk 
        """
        os.makedirs(dir_, exist_ok=True)

        fud_file = os.path.join(dir_,f"fud_{self.tag}.parquet")
        print(f"Writing {len(self.gdf):,} reviews as {fud_file}...")
        self.reviews.to_parquet(fud_file)

        print(f"Wrote '{self.tag}' dataset to {dir_}.")

    def load(self, dir_): 
        """
        Read our datasets off disk 
        """    
        reviews_path = os.path.join(dir_, f"reviews_{self.tag}.parquet")

        print(f"Loading reviews... ")
        gdf = pd.read_parquet(reviews_path) 

        print(f"Memory usage:{self.get_memory_usage()}")

    def split(self):
        """
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