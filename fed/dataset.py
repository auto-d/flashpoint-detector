import math 
import random
import pickle
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

#TODO: blend this into below skeletong and finish the dataset abstraction 

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
        self.spatial_step = 0.05
        self.spatial_step_km = 111 * self.spatial_step

        # Day increment to subdividide our temporal dimension by 
        self.temporal_step = temporal_step

        # Looking at the distribution of conflict event types, I'm not sure there's value in distinguishing a 
        # shelling/artillery/missile attack and an armed clash, for example. While ACLED makes a 
        # distinction, I think the extra complexity of juggling these labels isn't useful. 

        # Russia invaded Ukraine during the morning of the 24th of February, 2022. However this *relatively* 
        # quiescent period was still marked by skirmishes in the Donbas by Russia-backed separatist groups. 
        # These clashes intensified circa February of 2022, creating a pretext for the imminent Russian 
        # invasion. We'll mask all data during this 'transition' (from low-intensity to high-intensity conflict) 
        # period and consider the 24th the first day of escalated belligerence and overt war. 
        self.transition_period = (pd.Timestamp(year=2022, month=2, day=1), pd.Timestamp(year=2022, month=2, day=23))

    def download_fud(self, force=False):
        """
        Retrieve the latest version of the dataset from its home on kaggle, 
        optionally forcing the download regardless of whether we have a cached copy. 
        """
        tqdm.write(f"Retrieving FUD from Kaggle (force={force})...")
        self.path = kagglehub.dataset_download("justanotherjason/flashpoint-ukraine-dataset", path="fed.gpkg", force_download=force)

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
        tqdm.write(lon_steps, lat_steps, time_steps)

        self.quiescent_days = (self.transition_period[0] - self.min_date).days
        self.conflict_days = (self.max_date - self.transition_period[1]).days 

        # Conjure a massive numpy multidimensional array to back all future computations
        self.lattice = np.zeros((lon_steps, lat_steps, time_steps, len(self.feature_ixs)))

        gdf['conf_d'] = gdf.f_conf.apply(lambda x: 0.25 if x == 'l' else 0.5 if x == 'n' else 0.75)
        gdf['night_d'] = gdf.f_daynight.apply(lambda x: 0 if x == 'D' else 1)
        
        tqdm.write(f"Discretizing dataset at {self.spatial_step} decimal degree ({self.spatial_step_km:.2f} km) and "\
                   "{self.temporal_step} day resolution. This will create a grid of {lon_steps} x {lat_steps} x {time_steps} "\
                   "({lon_steps * lat_steps * time_steps:,} cells).")
        tqdm.write(f"Training aperture will be {self.story_width} steps square ({self.story_width * self.spatial_step_km:.2f} km"\
                   "total and {self.story_depth} steps long ({self.story_depth} days)).")

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
            self.lattice[x, y, d][self.feature_ixs['label']] = label
            self.lattice[x, y, d][self.feature_ixs['count']] = count + 1
        
        # Now generate the basic training/test/validation objects 
        self.permute_stories()

    def randomize_coords(self, x, x_max, y, y_max): 
        x = random.randrange(max(0, x - self.story_width), min(x_max, x + self.story_width))
        y = random.randrange(max(0, y - self.story_width), min(y_max, y + self.story_width))

        return x, y
    
    def permute_stories(self):         
        """
        Given the raw numpy dataset, discover and memorialize our 'stories' (contiguous blocks of time and space
        that are within the bounds of the dataset)
        """
        self.stories = np.zeros((self.n_quiescent_stories + self.n_conflict_stories, 3), dtype=np.int32)

        x_max = self.lattice.shape[0]
        y_max = self.lattice.shape[1]

        # Boundaries for the negative class - we must have enough room for priors and we have to cut off by the transition 
        n_start = 0 + self.story_depth
        n_end = ((self.transition_period[0]) - self.min_date).days

        # Boundaries for the positive class - must start after the transition and end before we run out of data!
        p_start = ((self.transition_period[1]) - self.min_date).days
        p_end = (self.max_date - self.min_date).days

        tqdm.write(f"Sampling {self.n_quiescent_stories} stories from the pre-war period ({self.min_date} - {self.transition_period[0]},"\
                   "{self.quiescent_days} days total) and {self.n_conflict_stories} stories from the post-invasion period "\
                   "{self.transition_period[1]} - {self.max_date}.")

        ixs = np.nonzero(self.lattice[:, :, :, self.feature_ixs['label']])

        quiescent = 0 
        conflict = 0 

        # Sample stories with variation in the spatial aperture until our counters are full.  This 
        # accomplishes class balance and allows scaling to arbitrary dataset size. 
        with tqdm(total = self.n_quiescent_stories + self.n_conflict_stories) as progress: 
            while quiescent + conflict < self.n_quiescent_stories + self.n_conflict_stories: 

                i = random.randrange(0, len(ixs[0]))
                x = ixs[0][i]
                y = ixs[1][i]
                t = ixs[2][i]
                candidate = self.lattice[x,y,t]

                # TODO: use 0 and np.nan here to avoid later conversions
                if candidate[self.feature_ixs['label']] == -1:
                    if quiescent < self.n_quiescent_stories: 
                        if t >= n_start and t < n_end: 
                            x, y = self.randomize_coords(x, x_max, y, y_max)            
                            self.stories[quiescent + conflict] = [x,y,t]
                            quiescent += 1
                            progress.update(1)
                
                elif candidate[self.feature_ixs['label']] == 1:
                    if conflict < self.n_conflict_stories: 
                        if t >= p_start and t < p_end: 
                            x, y = self.randomize_coords(x, x_max, y, y_max)                
                            self.stories[quiescent + conflict] = [x,y,t]
                            conflict += 1
                            progress.update(1)
                else: 
                    raise ValueError(f"Unknown label encountered when selecting stories: {candidate[7]}!")            
        
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
        val_count = val if val >= 1.0 else val * int(len(self.stories))
        test_count = test if test >= 1.0 else test * int(len(self.stories))
        
        # Build a list of all indicies, then remove them from the list as we sample
        # We'll avoid breaking up our core dataset here and instead just pass a list of indicies
        story_ixs = np.arange(0, len(self.stories)) 
        self.val = random.choices(story_ixs, k=val_count)
        np.delete(story_ixs, self.val)

        self.test = random.choices(story_ixs, k=test_count)
        np.delete(story_ixs, self.test)

        self.train = story_ixs

        # TODO: clean all test stories with intersect_stories and invent a solution to the 
        # lost index problem 

    def get_story(self, ix): 
        return Story(
            depth=self.story_depth,
            width=self.story_width, 
            x=self.stories[ix][0], 
            y=self.stories[ix][1], 
            t=self.stories[ix][2], 
        )
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
        dense = self.lattic[
            story.x : story.x + story.width, 
            story.y : story.y + story.width, 
            story.depth-1,
            self.feature_ixs['label']
            ]
        return dense 

    def split(self):
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
        tqdm.write(f"Splitting dataset ({len(self.gdf)} rows)")
        
        self.partition_stories()
        
        tqdm.write(f"Done! Post-split counts:\n"\
                   f" - ({len(self.train)} training stories)"\
                   f" - ({len(self.val)} val stories)"\
                   f" - ({len(self.test)} test stories)"\
                )
            
    def store(self, dir_):
        """ 
        Write the FED dataset to disk 
        """
        os.makedirs(dir_, exist_ok=True)
    
        path = os.path.join(dir_,f"fed_{self.tag}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        tqdm.write(f"Wrote '{self.tag}' FED dataset to {path}.")

    def load(self, dir_): 
        """
        Load the dataset off disk 
        """
        path = os.path.join(dir_, f"fed_{self.tag}.pkl")

        print(f"Loading FED... ")

        # TODO: this is probably not going to work see how big the saved pickled array is above and then 
        # decide how to proceed
        # with open(path, 'rb') as f: 
        #     return(pickle.load(f))
    
        # if type(state) != CfnnEstimator: 
        #     raise ValueError(f"Unexpected type {type(model)} found in {filename}")     

    def make_box(x=30.0, y=47.0, z=0, w=2, h=4):
        """
        Create a box to illustrate our spatiotemporal cross-validation strategy

        NOTE: setup of box geometry suitable for matplotlib courtesy of gpt-4o, part of the 
        larger conversation on rendering spatial data with matplotlib et al: 
        https://chatgpt.com/share/688cddab-ae9c-8013-ac1a-6b9211e72971
        """ 

        # Make boundaries 
        x0, x1 = x, x+w
        y0, y1 = y, y+w
        z0, z1 = z, z+h

        # Build the corners 
        corners = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]

        # Make assoc'd faces 
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[1], corners[2], corners[6], corners[5]],  # right
            [corners[0], corners[3], corners[7], corners[4]],  # left
        ]

        return faces 
    
    def render_box(box): 
        """
        Helper to visualize a box constructed by make_box
        """                
        block_color = (1, 0, 0, 0.3)  # semi-transparent red, courtesy of gpt-4o

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        box = Poly3DCollection(box, facecolors=(1, 0, 0, 0.3), edgecolors='r')
        ax.add_collection3d(box)

        # Set axis limits and labels
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        

    def set_axes_range(ax): 
        """
        We need to adjust the axes range to keep things proportional, matplotlib doesn't do this by 
        default. 
        NOTE: This hack courtesy of gpt-4o. See link in make_box()
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_zlim(z_limits[0], z_limits[1])        

    def temporal_scatter(gdf, n_dates=10):
        """
        Create a scatter plot of x,ys (presumed to be the active geometry in the supplied DF)
        using a date column. 

        NOTE: Geopandas and matplotlib maneuvering with help from gpt-4o: 
        https://chatgpt.com/share/688cddab-ae9c-8013-ac1a-6b9211e72971
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        dates = sorted(gdf['date'].unique())
        for i, date in enumerate(dates):
            if i >= n_dates:
                break 

            slice_df = gdf[gdf['date'] == date]
            xs = slice_df.geometry.x
            ys = slice_df.geometry.y
            zs = np.full_like(xs, i)
            ax.scatter(xs, ys, zs, label=f'Date {date}', depthshade=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Date')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_native_dataset(gdf, n_dates=1): 
        temporal_scatter(gdf, n_dates=2)

    def temporal_scatter_w_poly(gdf, geom, n_dates=3, color_map='OrRd', box=None, edge='r'): 
        """
        Plot x/y coordinates for date groupings with reference geometry for the 
        first n dates (oldest to newest). Presumes we have a 'date' column in the 
        supplied df. Also accepts a set of faces to plot in the box param, constructed
        ideally w/ make_box(). 
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        dates = sorted(gdf['date'].unique())
        
        cmap = cm.get_cmap(color_map, n_dates)
        norm = mcolors.Normalize(vmin=0, vmax=n_dates)
        ix_to_color = {t: cmap(norm(t)) for t in range(n_dates)}
        
        for i, date in enumerate(dates):    
            if i >= n_dates:
                break 
            
            color = ix_to_color[i]
        
            # Plot polygons associated with whatever geometry we've been given    
            polys = list(geom.geoms)
        
            for poly in polys:
                x, y = poly.exterior.xy
                z = [i] * len(x)
                verts = [list(zip(x, y, z))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolor=color))
            
            # Slice DF along dates
            date_df = gdf[gdf['date'] == date]
        
            # Plot associated detections     
            xs = date_df.location.x
            ys = date_df.location.y
            zs = np.full_like(xs, i)
            ax.scatter(xs, ys, zs, color=color, label=f'Date {date}', depthshade=True)
        
        # If we've been given a prismbox to render, shim it in there
        if box is not None: 
            block_color = (1, 0, 0, 0.3)  # semi-transparent red, courtesy of gpt-4o
            prism = Poly3DCollection(box, facecolors=block_color, edgecolors=edge)
            ax.add_collection3d(prism)

        # Adjust the extents to ensure scale in the x/y and zoom to the polygon provided... 
        xmin, ymin, xmax, ymax = geom.bounds
        x_span = xmax-xmin
        y_span = ymax-ymin 

        # if x_span > y_span: 
        #     ax.set_xlim(xmin, xmax)
        #     ax.set_ylim(ymin-x_span/2, ymax+x_span/2)
        # elif y_span > x_span: 
        #     ax.set_xlim(xmin-y_span/2, xmax+y_span/2)
        #     ax.set_ylim(ymin, ymax)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.set_zlabel('Date')
        plt.legend()
        ax.set_zlim(0, n_dates + 1)
        plt.tight_layout()
        plt.show()        

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