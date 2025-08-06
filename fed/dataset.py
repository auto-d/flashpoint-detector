import math 
import random
import pickle
import os
import pandas as pd
import numpy as np 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import kagglehub
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch 
from . import similarity

class Story():
    """
    Abstract the notion of a 'story', which is a contiguous spatiotemporal block of feature data, culminating (temporally) 
    in one or more labeled conflict events we can use for training or validation.
    """

    def __init__(self, depth, width, t=0, x=0, y=0,type='train'): 
        """
        Create a new story! 
        """
        self.depth = depth 
        self.width = width 

        self.t = t        
        self.x = x 
        self.y = y 
        
        self.type = type

    def __str__(self): 
        return f"t: {self.t}\nx:{self.x}\ny:{self.y}"

class FlashpointsDataset():     
    """
    Abstraction for the Flashpoints Ukraine Dataset (FUD) and associated pipeline. This ingests and 
    manipulates the FUD and stores the Flashpoints Event Detector (FED) dataset used for modeling. 
    """

    # Input features of the thermal anomaly we'll drive the prediction with 
    feature_ixs = {
        'bright1': 0,  
        'bright2': 1,    
        'power': 2, 
        'night': 3, 
        'conf': 4, 
        'month': 5, 
        'count' : 6,
        'label' : 7
    }

    def __init__(
            self, 
            tag, 
            story_width=20, 
            story_depth=7, 
            spatial_step=0.05, 
            temporal_step=1, 
            n_quiescent_stories = 1000000, 
            n_conflict_stories = 1000000
            ): 
        """
        Set up the instance, deferring creation/load given the memory-intensive nature of the operation
        """
        self.tag = tag

        # These are runtime flags to dataset, which are memorialized and used during 
        # training to ensure alignment and checked during testing to avoid exceptions during prediction
        self.story_width = story_width
        self.story_depth = story_depth

        # Lat/lon increment to subdivide our spatial dimension by
        self.spatial_step = spatial_step
        self.spatial_step_km = 111 * self.spatial_step

        # Day increment to subdividide our temporal dimension by 
        self.temporal_step = temporal_step

        self.n_quiescent_stories = n_quiescent_stories
        self.n_conflict_stories = n_conflict_stories

        # Looking at the distribution of conflict event types, I'm not sure there's value in distinguishing a 
        # shelling/artillery/missile attack and an armed clash, for example. While ACLED makes a 
        # distinction, I think the extra complexity of juggling these labels isn't useful. 

        # Russia invaded Ukraine during the morning of the 24th of February, 2022. However this *relatively* 
        # quiescent period was still marked by skirmishes in the Donbas by Russia-backed separatist groups. 
        # These clashes intensified circa February of 2022, creating a pretext for the imminent Russian 
        # invasion. We'll mask all data during this 'transition' (from low-intensity to high-intensity conflict) 
        # period and consider the 24th the first day of escalated belligerence and overt war. 
        self.transition_period = (pd.Timestamp(year=2022, month=2, day=1), pd.Timestamp(year=2022, month=2, day=23))

        self.scaled = False

    def download_fud(self, force=False):
        """
        Retrieve the latest version of the dataset from its home on kaggle, 
        optionally forcing the download regardless of whether we have a cached copy. 
        """
        tqdm.write(f"Retrieving FUD from Kaggle (force={force})...")
        self.path = kagglehub.dataset_download("justanotherjason/flashpoint-ukraine-dataset", path="fud.gpkg", force_download=force)

        tqdm.write(f"Done. Path to local dataset files: {self.path}")

    def load_fud(self):
        """
        Load the dataset into memory 
        """
        self.download_fud()

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

        tqdm.write(f"Loaded FUD, with coordinate system {gdf.crs}")  
        return gdf        

    def build(self, admin_boundaries_path): 
        """
        Build the dataset based on the raw FUD data and a geospatial reference
        """

        # Import administrative boundaries to anchor the discretization, we want to ensure the 
        # entire conflict area is represented from the get-go. Layer 7 is the country-level multipolygon. 
        admin = gpd.read_file(admin_boundaries_path, layer=7)

        min_lon, min_lat, max_lon, max_lat = admin.iloc[0].geometry.bounds
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat

        # Load/retrieve the flashpoints dataset
        gdf = self.load_fud()
        self.max_date = gdf.date.max()
        self.min_date = gdf.date.min()
        time_range = (self.max_date - self.min_date).days

        lon_steps = int(math.floor(lon_range/self.spatial_step)) + 1
        lat_steps = int(math.floor(lat_range/self.spatial_step)) + 1
        time_steps = int(math.floor(time_range/self.temporal_step)) + 1
        tqdm.write(f"Longitude steps @ {lon_steps}, latitude @ {lat_steps}, and time @ {time_steps}")

        self.quiescent_days = (self.transition_period[0] - self.min_date).days
        self.conflict_days = (self.max_date - self.transition_period[1]).days 

        # Conjure a massive numpy multidimensional array to back all future computations
        self.lattice = np.zeros((lon_steps, lat_steps, time_steps, len(self.feature_ixs)))

        gdf['conf_d'] = gdf.f_conf.apply(lambda x: 0.25 if x == 'l' else 0.5 if x == 'n' else 0.75)
        gdf['night_d'] = gdf.f_daynight.apply(lambda x: 0 if x == 'D' else 1)
        
        tqdm.write(f"Discretizing dataset at {self.spatial_step} decimal degree ({self.spatial_step_km:.2f} km) and "\
                   f"{self.temporal_step} day resolution. This will create a grid of {lon_steps} x {lat_steps} x {time_steps} "\
                   f"({lon_steps * lat_steps * time_steps:,} cells).")
        tqdm.write(f"Training aperture will be {self.story_width} steps square ({self.story_width * self.spatial_step_km:.2f} km"\
                   f"total and {self.story_depth} steps long ({self.story_depth} days)).")

        # Discretize the area into cells and storing as a 'lattice' here to support our modeling aspirations
        for i, row in tqdm(gdf.iterrows(), total=len(gdf)): 
            
            # Scale in decimal degrees and days based on our step values
            x = int((row.f_lng - min_lon)/self.spatial_step) 
            y = int((row.f_lat - min_lat)/self.spatial_step)
            d = int((row.date - self.min_date).days/self.temporal_step)

            # Make a new feature vector
            label = -1 if pd.isnull(row.a_event) else 1
            feature = [row.f_bright, row.f_bright31, row.f_frp, row.night_d, row.conf_d, row.date.month, 1, label]

            # Aggregate. Ideally this only happens once and only a single feature gets recorded per 
            # cell, but collisions happen often enough at large cell sizes that we need a general strategy. 
            # Conflict events here are sticky, if it was seen once in the cell, it will always be part of the positive 
            # class. 
            count = self.lattice[x, y, d][self.feature_ixs['count']]
            self.lattice[x, y, d] = (self.lattice[x, y, d] * count + feature) / (count + 1)
            
            # TODO: check up on this label application to ensure we're carrying the right values forward
            self.lattice[x, y, d][self.feature_ixs['label']] = label
            self.lattice[x, y, d][self.feature_ixs['count']] = count + 1
        
        # Now generate the basic training/test/validation objects 
        self.permute_stories()

    def randomize_coords(self, x, y):
        """
        Given an x and y location for a labeled event, find a random slice of the spatial dimension 
        which still includes that location (and its label). Recall the story we'll record is anchored at the 
        bottom/low end of each dimension and runs story_width wide. 
        """
        w = self.story_width

        x = random.randrange(max(0, x - w + 1),max(x,1))
        y = random.randrange(max(0, y - w + 1),max(y,1))

        # If the story is off the board, clamp it to allow a full-width story
        if x + w > self.x_max: 
            x = self.x_max - w
        if y + w > self.y_max: 
            y = self.y_max - w 

        if x + self.story_width > self.x_max or y + self.story_width > self.y_max: 
            raise ValueError("Story out of range! Perhaps fewer steps available than the story width requires.")
    
        return x, y
    
    def validate_story(self, story=None, ix=None): 
        """
        Sanity check to detect wayward stories at creation or training time 
        """
        
        if ix is not None: 
            story = self.get_story(ix)
            
        # In bounds?
        assert(story.x <= self.x_max) 
        assert(story.y <= self.y_max) 
        assert(story.t >= 0 and story.t <= self.n_end or\
                story.t >= self.p_start and story.t <= self.p_end)
        
        # Concludes with one or more labels? 
        labels = self.label_story(story)
        values = np.unique(labels)
        assert(-1. in values or 1 in values)

    def permute_stories(self):         
        """
        Given the raw numpy dataset, discover and memorialize our 'stories' (contiguous blocks of time and space
        that are within the bounds of the dataset). Every story will have at least one label in its ultimate time
        step. The tuple x,y,t identify a story uniquely, where: 
         - x = the lower bound of the longitude/x region
         - x + story_width = upper bound of the long/x region 
         - y = the lower bound of the region in the latitude/y dimension 
         - y + story_width = upper bound of the lat/y
         - t = lower bound in the temporal dimension 
         - t + story_depth = upper bound (end) of the story in the temporal dim 
        """
        self.stories = np.zeros((self.n_quiescent_stories + self.n_conflict_stories, 3), dtype=np.int32)

        self.x_max = self.lattice.shape[0]
        self.y_max = self.lattice.shape[1]

        # Boundaries for the negative class - we must have enough room for priors and we have to cut off by the transition 
        self.n_start = 0 + self.story_depth
        self.n_end = ((self.transition_period[0]) - self.min_date).days

        # Boundaries for the positive class - must start after the transition and end before we run out of data!
        self.p_start = ((self.transition_period[1]) - self.min_date).days
        self.p_end = (self.max_date - self.min_date).days

        tqdm.write(f"Sampling {self.n_quiescent_stories} stories from the pre-war period ({self.min_date} - {self.transition_period[0]},"\
                   f"{self.quiescent_days} days total) and {self.n_conflict_stories} stories from the post-invasion period "\
                   f"{self.transition_period[1]} - {self.max_date}.")

        # Only allow windows that end with a non-zero label (-1 or 1)... ensures all stories 
        # have an ending. 
        ixs = np.nonzero(self.lattice[:, :, :, self.feature_ixs['label']])

        quiescent = 0 
        conflict = 0 

        # Sample labeled points with variation in the spatial aperture until our counters are full.  This 
        # accomplishes class balance and allows scaling to arbitrary dataset size. 
        with tqdm(total = self.n_quiescent_stories + self.n_conflict_stories) as progress: 
            while quiescent + conflict < self.n_quiescent_stories + self.n_conflict_stories: 

                i = random.randrange(0, len(ixs[0]))
                x = ixs[0][i]
                y = ixs[1][i]
                t = ixs[2][i]
                candidate = self.lattice[x,y,t]

                # Create a story for every valid labels that leaves us enough room for our 
                # story brick (dim x,y,t)
                if candidate[self.feature_ixs['label']] == -1:
                    if quiescent < self.n_quiescent_stories: 
                        if t - self.story_depth + 1 >= self.n_start and t < self.n_end: 
                            x2, y2 = self.randomize_coords(x, y)            
                            self.stories[quiescent + conflict] = [x2,y2,t-self.story_depth+1]
                            self.validate_story(ix=quiescent + conflict)
                            quiescent += 1
                            progress.update(1)
                
                elif candidate[self.feature_ixs['label']] == 1:
                    if conflict < self.n_conflict_stories: 
                        if t - self.story_depth + 1 >= self.p_start and t < self.p_end: 
                            x2, y2 = self.randomize_coords(x, y)                                            
                            self.stories[quiescent + conflict] = [x2,y2,t-self.story_depth+1]
                            self.validate_story(ix=quiescent + conflict)           
                            conflict += 1
                            progress.update(1)
                else: 
                    raise ValueError(f"Unknown label encountered when selecting stories: {candidate[self.feature_ixs['label']]}!")

        tqdm.write("Generation complete!")    
        
    def plot_story(self, story, start=None, end=None, heatmap=False):
        """
        Plot a map of events from our 4d story lattice given a time range to 
        subset on
        """
        counts = None 
        if start is not None and end is not None: 
            counts = story[:, :, start:end, self.feature_ixs['count']]
        else: 
            counts = story[:, :, :, self.feature_ixs['count']]
        
        counts = np.sum(counts, axis=2)
        x0, y0 = np.nonzero(counts)
        values = counts[x0,y0]
        
        kwargs = { 'c': values, 'cmap': 'magma' } if heatmap else {}

        plt.scatter(x0, y0, **kwargs)
        
    def flatten_stories(self, ixs):
        """
        Flatten provided stories using feature averaging. 
        
        We started with a rather nicely curated geodataframe and had to mine it
        to create our spatiotemporal bricks ('stories') to power a deeper analysis
        yet still achieve a fixed-width representation. The volume of this new data
        is problematic for classic models which want to eat all their data in one sitting
        Here we'll create a streamlined dataframe which balances the richness of the 
        new features against the challenges of pushing rich 4d data through a classic 
        estimator like a random forest.
        """
        flattened = np.zeros((len(ixs), self.story_width** 2 * (len(self.feature_ixs)-1)))
        for i, ix in tqdm(enumerate(ixs), total=len(ixs)): 
            story = self.get_story(ix)
            ds = self.densify_story(story) 
            
            # Flatten our features through averaging to make this approachable. Note this is a large
            # feature matrix (2800 columns in the 'small' configuration). Beware attempting
            # this at higher spatial resolutions accordingly. 
            flattened[i] = np.mean(ds, axis=2).flatten()
        return flattened
    
    def flatten_labels(self, ixs, categories=False):
        """
        As with story flattener above, we need to offer a flattened data type for models that 
        can't handle the dimensionality of what is now our native view (4d)
        """
        flattened = np.zeros((len(ixs), self.story_width** 2))
        for i, ix in tqdm(enumerate(ixs), total=len(ixs)): 
            story = self.get_story(ix)
            dl = self.label_story(story)
            
            flattened[i] = dl.flatten()
        
        flattened[flattened == 1] = 2
        flattened[flattened == -1] = 1
        return flattened
        
    def intersect_stories(self, story):
        """
        Find and remove any stories that interesect in time or space with the provided
        story. This supports validation and test strategies which we desire to avoid being 
        tainted by data from our training sets. 
        """

        # TODO: we need to implement a means for 
        intersection = (
            (self.stories[:,0]>=story[0]) & 
            (self.stories[:,0]<=(story[0] + self.story_width)) & 
            (self.stories[:,1]>=story[1]) & 
            (self.stories[:,1]<=(story[1] + self.story_width)) & 
            (self.stories[:,2]>=story[2]) & 
            (self.stories[:,2]<=(story[2] + self.story_depth))
            )
        
        # update the story list by dropping any offending stories (except the one that induced the update)
        self.stories[intersection] = -1

        # TODO: restore the target story ... need to track an index somewhere? 
        self.stories[4] = story 

        # Then drop all the other stories that were zeroed out
        self.stories = self.stories[~intersection]
        len(self.stories)

        # TODO: figure out how to avoid this crashing any cached indices

    def partition_stories(self, val=0.1, test=0.1):
        """
        Partition our stories into train, validation and test sets. Any stories not reserved for validation 
        or test are allocated to training. If value is less than one, it's treated as a percentage to 
        split out, otherwise it's a count of stories to hold out. 
        """
        val_count = val if val >= 1.0 else int(val * len(self.stories))
        test_count = test if test >= 1.0 else int(test * len(self.stories))
        
        # Build a list of all indicies, then remove them from the list as we sample
        # We'll avoid breaking up our core dataset here and instead just pass a list of indicies
        story_ixs = np.arange(0, len(self.stories)) 
        self.val = random.choices(story_ixs, k=val_count)
        story_ixs = np.delete(story_ixs, self.val)

        self.test = random.choices(story_ixs, k=test_count)
        story_ixs = np.delete(story_ixs, self.test)

        self.train = story_ixs

        # TODO: clean all test stories with intersect_stories and invent a solution to the 
        # lost index problem 

        return self.train, self.val, self.test

    def get_story(self, ix): 
        story =  Story(
            depth=self.story_depth,
            width=self.story_width, 
            x=self.stories[ix][0], 
            y=self.stories[ix][1], 
            t=self.stories[ix][2], 
        )

        if story.x + self.story_width > self.x_max or story.y + self.story_width > self.y_max: 
            raise ValueError("Story out of range! Perhaps fewer steps available than the story width requires.")
        
        return story 

    def scale_features(self): 
        """
        Irrevocably switch the dataset to scaled float mode, where all features are normalized to a float
        within the range [0,1]
        """
        
        if not self.scaled: 
            
            # Iterate over our feature indices and scale each by the max value 
            for ix in self.feature_ixs.values(): 
                self.lattice[:,:,:,ix] = self.lattice[:,:,:,ix] / np.max(self.lattice[:,:,:,ix])

        self.scaled = True 
    
    def densify_story(self, story): 
        """
        Retrieve a dense representation of the story, sans labels. 
        """

        # Leverage numpy slicing across dimensions to quickly extract a view into the story from the 
        # core array object, dropping the label from the feature vector. 
        dense = self.lattice[
            story.x : story.x + story.width, 
            story.y : story.y + story.width, 
            story.t : story.t + story.depth,
            0:self.feature_ixs['count'] + 1
            ]

        return dense

    def label_story(self, story): 
        """
        Retrieve a dense array of labels for the provided story
        """
        
        # Slice out the last day of features for this story (predicting the classification of a cell 
        # on the last day is always our target). Note this drops the third dimension, so we emit a 
        # width x width matrix of scalar labels. I.e. every cell has it's own label. 
        dense = self.lattice[
            story.x : story.x + story.width, 
            story.y : story.y + story.width, 
            story.t + story.depth-1,
            self.feature_ixs['label']
            ]
        
        return dense 

    def split(self, val=10, test=10):
        """
        Splitting data for training and evaluation in the context of a potentially self-exciting process as 
        we are dealing with in conflict events is problematic as: 
        1. prior conflict events influence future conflict events, doing a naive temporal or class-based split 
        both deprives us at prediction time of relevant priors and leaks ground-truth at training time 
        2. ignores the fact that latent patterns we're trying to predict may evolve over time, and a 
        temporal split would potentially remove a large swath of information a model might need to learn. 

        Drawing inspiration from weather prediction models (which have the same problem) we opt for a 
        floating 3d window that we randomly sample with replacement from the core dataset. We call these floating 
        window/block/box things 'stories', since they have an arc over time and have a locality element driven by 
        intuition that suggeststhere are potentially causal events in/around them.  Test sets will be selected
        from this 'story' population and any intersecting stories will be removed to avoid contamination of test results. 

        See conversation with gpt-4o on meteorological approaches: https://chatgpt.com/share/688a362f-7ba4-8013-a4b8-3be2073f8569 

        Generate splits as follows: 
        - self.train : training matrix
        - self.val : matrix to predict on during validation 
        - self.test : matrix for test predictions
        """
        tqdm.write(f"Splitting dataset with val @ {val}, test @ {test}... ")
        
        self.partition_stories(val=val, test=test)
        
        tqdm.write(f"Done! Post-split counts:\n"\
                   f" - ({len(self.train)} training stories)\n"\
                   f" - ({len(self.val)} val stories)\n"\
                   f" - ({len(self.test)} test stories)\n"\
                )
            
    def __getstate__(self):
        """
        Retrieve our state for saving

        NOTE: gpt-4o assist on syntax to allow pickling of object but custom storage of huge numpy ndarraysj:
        https://chatgpt.com/share/689297a2-c9bc-8013-8418-31a1adef2b42
        """
        state = self.__dict__.copy()
        
        # Zeroize the hefty arrays, we'll handle their storage separately...
        state['lattice'] = None
        state['stories'] = None
        return state

    def __setstate__(self, state):
        """
        Load state

        NOTE: gpt-4o assist on syntax to allow pickling of object but custom storage of huge numpy ndarraysj:
        https://chatgpt.com/share/689297a2-c9bc-8013-8418-31a1adef2b42
        """        
        self.__dict__.update(state)

    def store(self, dir_):
        """ 
        Write the FED dataset to disk, sadly this is a two-parter given the presence of large numpy ndarrays
        """
        os.makedirs(dir_, exist_ok=True)
    
        path = os.path.join(dir_,f"fed_{self.tag}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        path = os.path.join(dir_,f"fed_{self.tag}_lattice.npz")            
        np.savez_compressed(path, self.lattice)

        path = os.path.join(dir_,f"fed_{self.tag}_stories.npz")            
        np.savez_compressed(path, self.stories)

        tqdm.write(f"Wrote '{self.tag}' FED dataset to {dir_}.")

    @classmethod
    def load(self, dir_, tag): 
        """
        Load the dataset off disk 
        """
        path = os.path.join(dir_, f"fed_{tag}.pkl")

        tqdm.write(f"Loading FED from {dir_}... ")

        with open(path, 'rb') as f: 
            obj = pickle.load(f)
    
        if type(obj) != FlashpointsDataset: 
             raise ValueError(f"Unexpected type {type(obj)} found in {path}!")     
        
        path = os.path.join(dir_,f"fed_{tag}_lattice.npz")            
        with np.load(path, encoding='bytes') as data: 
            obj.lattice = data['arr_0']

        path = os.path.join(dir_,f"fed_{tag}_stories.npz")  
        with np.load(path) as data: 
            obj.stories = data['arr_0']

        return obj     

class FlashpointsTorchDataset(torch.utils.data.Dataset):
    """
    Torch-compatible dataset to capitalize on the former's abstraction of batching, 
    parallelism and shuffling memory to GPU & back. 

    NOTE: the shell of this class has been repurposed multiple times in prior assignments
    """

    def __init__(self, dataset:FlashpointsDataset, ixs, batch_size=10): 
        """
        Initialize a new instance given flashpoints dataset object and the in-bounds
        indices. Use of indicies avoids copying large numpy arrays around, in practice 
        it will be one of train, val, test subsets of the dataset's stories object. 
        """
        self.batch_size = batch_size
        self.ixs = ixs
        self.dataset = dataset
        
        # Irrevocable 
        dataset.scale_features()

    def __len__(self): 
        """
        Retrieve length of the dataset
        """
        return len(self.dataset.stories) 
    
    def __getitem__(self, idx): 
        """
        Retrieve an item at the provided index
        """
        story = self.dataset.get_story(self.ixs[idx])
        return self.dataset.densify_story(story)
    
    def get_data_loader(self, shuffle=True): 
        """
        Retrieve a pytorch-style dataloader that loads data with this instance
        """
        loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)        
        return loader