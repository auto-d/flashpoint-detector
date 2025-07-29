import sys 
import os
import pandas as pd
import numpy as np 
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.python_splitters import python_stratified_split 
from tqdm import tqdm 
import torch 

class DeepCartDataset(): 
    """
    Abstraction for the review-driven dataset that supports our recommenders. 
    """
    
    def __init__(self, tag): 
        """
        Set up the instance, deferring creation/load given the memory-intensive nature
        """
        self.tag = tag
        self.users = None
        self.reviews = None
        self.items = None

    def find_users(self, reviews): 
        """
        Given reviews, generate a user dataframe     
        """
        users = reviews.groupby(['user_id']).rating.count()
        users = pd.DataFrame(users).reset_index()
        users.rename(columns={'rating':'ratings'}, inplace=True)
        return users 

    def extract(self, items_path, reviews_path, min_interactions, min_ratings, sample_n): 
        """
        Given raw review and item data files, prepare a cleaned, live dataset. 
        """
        print(f"Extracing dataset from on {items_path} and {reviews_path}...")

        if not items_path.endswith(".parquet") or not reviews_path.endswith(".parquet"): 
            raise ValueError("Unexpected file type!")
        
        reviews = pd.read_parquet(reviews_path) 
        items = pd.read_parquet(items_path)    
        users = self.find_users(reviews)
        print(f"Found {len(users):,} users with {len(reviews):,} ratings of {len(items):,} items.")

        # Per dataset documentation we should always use parent_asin for correlations as 
        # children are just color, etc... variations of the same product
        reviews.rename(columns={'parent_asin':'item_id'}, inplace=True)
        items.rename(columns={'parent_asin':'item_id'}, inplace=True)

        # Products with few interactions are a weak signal -- we are looking to
        # connect users and items that have tiny interaction graphs are not going to improve
        # our macro-level predictions, but it will cost us in memory and compute
        popular_items = items[items.rating_number > min_ratings]        
        print(f"Dropped {len(items)-len(popular_items):,} unpopular items (<{min_ratings} ratings)")
        reviews_small = reviews[reviews.item_id.isin(popular_items.item_id.unique())]
        print(f"Dropped {len(reviews)-len(reviews_small):,} reviews (of unpopular items)")
        
        # Update our review counts to reflect only popular items
        users = self.find_users(reviews_small)

        # Users with few interactions are also a weak signal -- without associations with 
        # multiple products, we are not teaching the model about positive associations                
        active_users = users[users.ratings >= min_interactions]
        print(f"Dropped {len(users)-len(active_users):,} low-activity users (reviews <{min_interactions})")
        reviews_smaller = reviews_small[reviews_small.user_id.isin(active_users.user_id.unique())]
        print(f"Dropped {len(reviews)-len(reviews_small):,} reviews (of low-activity users)")

        # Update our review counts to only reflect active users and sample
        users = self.find_users(reviews_smaller)        
        sampled_users = users.sample(sample_n)
        self.reviews = reviews_smaller[reviews_smaller.user_id.isin(sampled_users.user_id)]
        print(f"Dropped {len(reviews_smaller)-len(self.reviews):,} reviews (user not part of sample)")

        # Finally clean all items that no longer have represntation in the reviews
        self.items = popular_items[popular_items.item_id.isin(self.reviews.item_id.unique())]    
        print(f"Dropped {len(popular_items)-len(self.items):,} items (no review associated)")
        
        print(f"Generation complete!")

    def store(self, dir_):
        """ 
        Write our dataset to disk 
        """
        os.makedirs(dir_, exist_ok=True)

        reviews_file = os.path.join(dir_,f"reviews_{self.tag}.parquet")
        print(f"Writing {len(self.reviews):,} reviews as {reviews_file}...")
        self.reviews.to_parquet(reviews_file)

        items_file = os.path.join(dir_,f"items_{self.tag}.parquet")
        print(f"Writing {len(self.items):,} items as {items_file}...")    
        self.items.to_parquet(items_file)

        print(f"Wrote '{self.tag}' dataset to {dir_}.")

    def load(self, dir_): 
        """
        Read our datasets off disk 
        """    
        reviews_path = os.path.join(dir_, f"reviews_{self.tag}.parquet")
        items_path = os.path.join(dir_, f"items_{self.tag}.parquet")

        print(f"Loading reviews... ")
        reviews = pd.read_parquet(reviews_path) 
            
        print(f"Loading items... ")
        items = pd.read_parquet(items_path)    
        
        print(f"Extracting users ... ")
        users = self.find_users(reviews)

        print(f"Memory usage:") 
        print(f" - reviews ~{sys.getsizeof(reviews):,} bytes)")
        print(f" - items ~{sys.getsizeof(items):,} bytes)")
        print(f" - users ~{sys.getsizeof(users):,} bytes)")

        self.reviews = reviews
        self.items = items 
        self.users = users         

    def split(self):
        """
        Generate sparse matrices for training splits as follows: 
        - self.train : training matrix
        - self.val : matrix to predict on during validation 
        - self.val_chk : matrix to check validation predictions on 
        - self.test : matrix for test predictions
        - self.test_chk : matrix to compare test predictions against

        To access the matrx, call gen_affinity_matrix() on each of the above, which will 
        densify (watch out for memory issues) and return the contiguous array of values. 
        [0] is the first user in the list, with ratings for all items in that row
        """
        tqdm.write(f"Full user-item matrix is {len(self.users) * len(self.items)}")

        # NOTE: Strategy adapted from tutorials available in the Recommenders project, see 
        # https://github.com/recommenders-team/recommenders/tree/main
        # Split along user boundaries to ensure no leakage of preference between train and test
        train_users, test_users, val_users = python_random_split(self.users, [.8, .1, .1])

        train = self.reviews[self.reviews.user_id.isin(train_users.user_id)]
        val = self.reviews[self.reviews.user_id.isin(val_users.user_id)]
        test = self.reviews[self.reviews.user_id.isin(test_users.user_id)]

        # Technique from Recommenders (see https://github.com/recommenders-team/recommenders/blob/45e1b215a35e69b92390e16eb818d4528d0a33a2/examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb) 
        # to improve utility of validation set during training - only allow items in
        # the validation set that are also present in the train set
        tqdm.write("Dropping items from validation set that are not present in train set...")
        val = val[val.item_id.isin(train.item_id.unique())]

        # Another technique employed in Recommenders (see above link for notebook), for in-flight validation to be 
        # meaningful during training, our validation set needs not just ground truth, but unseen validation samples 
        # to see if predictions for validation users are relevant (to those users). Anyway, break down our val and test 
        # sets again to support this strategy
        val_src, val_target = python_stratified_split(
            data=val, 
            ratio=0.7, 
            filter_by="item", 
            col_user="user_id", 
            col_item="item_id"
            )
        test_src, test_target = python_stratified_split(
            data=test, 
            ratio=0.7, 
            filter_by="item", 
            col_user="user_id", 
            col_item="item_id"
            )        
                
        tqdm.write(f"User splits:\n"\
            f" train : {len(train_users)} users @ {len(train)} ratings\n"\
            f" val   : {len(val_users)} users @ {len(val)} ratings\n"\
            f"  - input : {len(val_src)} ratings\n"\
            f"  - check : {len(val_target)} ratings\n"\
            f" test  :{len(test_users)} users @ {len(test)} ratings\n"\
            f"  - input :{len(test_src)} ratings\n"\
            f"  - check :{len(test_target)} ratings\n"\
            )

        header = {
            "col_user": "user_id",
            "col_item": "item_id",
            "col_rating": "rating",
            "col_pred" : "recs"
        }

        self.train = AffinityMatrix(df=train, **header)
        self.val = AffinityMatrix(df=val_src, **header)
        self.val_chk = AffinityMatrix(df=val_target, **header)
        self.test = AffinityMatrix(df=test_src, **header)
        self.test_chk = AffinityMatrix(df=test_target, **header)

class DeepCartTorchDataset(torch.utils.data.Dataset):
    """
    Torch-compatible dataset to capitalize on the former's abstraction of batching, 
    parallelism and shuffling memory to GPU & back
    """

    def __init__(self, matrix:AffinityMatrix=None, ui=None, u_map=None, i_map=None, batch_size=10): 
        """
        Initialize a new instance given a sparse matrix of reviews or the raw data (user-item matrix, 
        user map, item map).        
        """
        self.batch_size = batch_size
        if matrix: 
            self.ui, self.u_map, self.i_map = matrix.gen_affinity_matrix()
        elif ui is not None and u_map is not None and i_map is not None: 
            self.ui = ui
            self.u_map = u_map
            self.i_map = i_map 
        else: 
            raise ValueError("Can't construct dataset due to params!")

    def __len__(self): 
        """
        Retrieve length of the dataset
        """
        return len(self.ui) 
    
    def __getitem__(self, idx): 
        """
        Retrieve an item at the provided index
        """
        # Scale reviews to [0,1] for our network 
        reviews = self.ui[idx]
        item = np.divide(reviews, 5).astype(np.float32)
        return item 

    def get_mappings(self): 
        """
        Retrieve the user and item mappings to enable decoding of the user-item 
        affinity matrix 
        """
        return self.u_map, self.i_map 
    
    def get_data_loader(self, shuffle=True): 
        """
        Retrieve a pytorch-style dataloader that loads data with this instance
        """
        loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)        
        return loader