import gradio as gr
import pandas as pd
import numpy as np
from .dataset import FlashpointsDataset, FlashpointsTorchDataset
from . import nn
from . import similarity
from gradio_folium import Folium
from folium import Map
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dataset = None
test = None
gdf = None
admin = None 
model = None 

def initialize(tag):         
    """
    Initialize global state and populate initial recommendation s
    """
    global test
    global gdf 
    global admin 
    global model 
    global dataset
    
    dataset = FlashpointsDataset.load("data", tag=tag)
    gdf = dataset.load_fud()    
    dataset.split()
    test = FlashpointsTorchDataset(dataset, dataset.val, batch_size=1)

    ukraine_admin_gdb = "data/ukr_admbnd_sspe_20240416_AB_GDB.gdb"    
    admin = gpd.read_file(ukraine_admin_gdb, layer=7)
    
    # Recover our model and create an empty dataset for our user(s)
    model = nn.load_model("models/") 

def select(df, data: gr.SelectData):
    row = df.iloc[data.index[0], :]
    return Map(location=[row['Latitude'], row['Longitude']])

def make_box(x=30.0, y=47.0, z=0, w=2, h=4):
    """
    Create a box to illustrate our spatiotemporal cross-validation (STCV) strategy 

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
    ax.axis('off')
    plt.tight_layout()

    return fig

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
    ax.axis('off')
    plt.legend()
    plt.tight_layout()
    
    return fig

def temporal_scatter_w_poly(geom, n_dates=3, color_map='OrRd', box=None, edge='r'): 
    """
    Plot x/y coordinates for date groupings with reference geometry for the 
    first n dates (oldest to newest). Presumes we have a 'date' column in the 
    supplied df. Also accepts a set of faces to plot in the box param, constructed
    ideally w/ make_box(). 
    """
    global gdf

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    dates = sorted(gdf['date'].unique())
    
    cmap = cm.get_cmap(color_map, n_dates)
    norm = mcolors.Normalize(vmin=0, vmax=n_dates)
    ix_to_color = {t: cmap(norm(t)) for t in range(n_dates)}
    
    for i, date in enumerate(dates[700:750]):    
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
        xs = date_df.a_lng
        ys = date_df.a_lat
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
    ax.axis('off')
    plt.legend()
    ax.set_zlim(0, n_dates + 1)
    plt.tight_layout()
    
    return fig
    
def plot_story(lattice, start=None, end=None, heatmap=False):
    """
    Plot a map of events from our 4d story lattice given a time range to 
    subset on
    """

    global dataset 

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()    

    counts = None 
    if start is not None and end is not None: 
        counts = lattice[:, :, start:end, 6]
    else: 
        counts = lattice[:, :, :, 6]
    
    # TODO: add label-specific colors? 
    #if len(story[0,0,0]) - 1 >= feature_ixs['label']:         
    
    counts = np.sum(counts, axis=2)
    x0, y0 = np.nonzero(counts)
    values = counts[x0,y0]
    
    kwargs = { 'c': values, 'cmap': 'magma' } if heatmap else {}

    ax.scatter(x0, y0, **kwargs)
    ax.axis('off')

    return fig

def plot_preds(lattice, start=None, end=None, story_ix=None, preds=None, heatmap=False):
    """
    Plot a map of events from our 4d story lattice given a time range to 
    subset on and some predictions
    """

    global dataset 

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()    

    counts = None 
    if start is not None and end is not None: 
        counts = lattice[:, :, start:end, 6]
    else: 
        counts = lattice[:, :, :, 6]
    
    counts = np.sum(counts, axis=2)
    x0, y0 = np.nonzero(counts)
    values = counts[x0,y0]
    
    kwargs = { 'c': values, 'cmap': 'magma' } if heatmap else {}

    ax.scatter(x0, y0, **kwargs)
        
    if preds is not None:         
        
        # Back to square... and offset by story
        preds = preds.reshape((20,20))
        x,y,t = dataset.stories[story_ix][0]

        x1 = []
        y1 = []
        
        # Offset the preds by the story
        for xb in range(0,20):            
            for yb in range(0,20):                
                if preds[xb,yb] > 0.5: 
                    x1.append(x+xb)
                    y1.append(y+yb)
                
        
        ax.scatter(x1,y1, marker='*', color='red')

    ax.axis('off')

    return fig

def render_story(): 
    global gdf

    return render_box(make_box(2,3,0,3,5))

def render_points(steps=3): 
    global gdf 
    global admin 

    return temporal_scatter_w_poly(admin.iloc[0].geometry, n_dates=steps, color_map="RdYlBu")

def render_points_w_story(steps=6): 
    global gdf 
    global admin 

    box = make_box(33.0,45.0, 3, 1, 2)
    
    return temporal_scatter_w_poly(
        admin.iloc[0].geometry, 
        n_dates=steps, 
        color_map="RdYlBu", 
        box=box)

def render_map(): 
    """
    Plot a basic map of ukraine for reference 
    """
    global admin     

    fig, ax = plt.subplots(figsize=(4, 4))
    admin.plot(ax=ax)
    ax.axis('off')
    
    return fig 

def update_timestep(step): 

    return render_points(step)

def show_story_view(): 
    global dataset

    return plot_story(dataset.lattice, start=0, end=100, heatmap=True)

def show_story_predictions(): 

    global dataset
    global date
    global model 

    if model is not None: 
        story_ix = [700]
        story_ds = FlashpointsTorchDataset(dataset, story_ix, batch_size=1)
        
        preds = model.predict(story_ds)
        preds = preds/preds.max()

        return plot_preds(dataset.lattice, start=0, end=100, story_ix=story_ix, preds=preds, heatmap=True)

def demo(share=False, data_tag="test"): 
    """
    Our Gradio demo app!

    NOTE: with snippets from folium, gradio and gradio_folium documentation 
    """
    global gdf
    global map 

    initialize(data_tag)         

    with gr.Blocks() as demo: 

        # Header         
        gr.Markdown(value="# 💥 Flashpoint Event Detector, Explained")
        gr.Markdown(value="##  Ukraine conflict event detection through NASA thermal anomaly data analysis.")
        gr.Markdown(value="The flashpoint ukraine detector leverages four years of thermal anomaly data from multiple NASA hyperspectral imagers contextualized by the Armed Conflict & Location Data (ACLED) source. Its purpose is to classify new thermal anomalies to disambiguate conflict events from other thermal sources such as naturally occuring forest fires.")

        gr.Markdown(value="This application provides an overview of the modeling and demonstrates the prediction feature. ")
        
        # Basemap 
        gr.Markdown("# 🗺️ Study Area")
        with gr.Row():
            with gr.Column(scale=2): 
                gr.Markdown(value="The study area is the entirety of the internationally recognized borders of Ukraine")
            with gr.Column(scale=1):
                basemap = gr.Plot(label="Ukraine", container=False,)

        # FUD points
        gr.Markdown("# 📍 Flashpoints Events")
        with gr.Row():                             
            with gr.Column(scale=1): 
                gr.Markdown(value="The Flashpoints Ukraine Dataset (FUD) labels NASA thermal anomalies where their geometry intersects with ACLED-reported conflict events. We elect to operate on a granularity of 1 or more days, depending on dataset build settings. Higher resolutions in the temporal dimension create issues given satellite overflight patterns and the maximum resolution available in our event reporting.\n\nThe full FUD spans roughly 4 years, covering a quiescent period in the conflict and the full-scale invasion and subsequent belligerence in 2022.")
            with gr.Column(scale=2): 
                time_slider = gr.Slider(label="Time slices to render", minimum=1, value=3, maximum=10, step=1)                    
                fud = gr.Plot(label="Flashpoints Ukraine Dataset slices") 
                time_slider.change(fn=update_timestep, inputs=time_slider, outputs=[fud])

        # Story 'splainer
        gr.Markdown("# 📗 Story Time")
        with gr.Row():             
            with gr.Column(scale=2): 
                gr.Markdown(value="A significant challenge exists in incorating a temporal dimension to an already complex geospatial dataset. We elect to discretize the spatiotemporal volume into blocks that ensure consistent dimensions for model training as well as implicitly support our cross validation strategy. This balances competing demands of untainted holdouts for testing and the need to reduce blind-spots in the latent patterns we unearth during training. We permute as many 3-dimensional regions as are required for meet our class targets and sample ensuring 100% balance between the positive and negative classes. ")
            with gr.Column(scale=1): 
                story = gr.Plot(label="Story geometry") 
                        
        # Story in the context of points 
        with gr.Row(): 
            with gr.Column(scale=2):
                story_context = gr.Plot(label="Story geometry in the context of sliced thermal anomalies")
            
        # Predictions
        gr.Markdown("# 💡 Event Predictions")
        with gr.Row(): 
            with gr.Column(): 
                gr.Markdown(value="The underlying dataset supports predictions during the relative quiescent period from September 1st, 2020 to February 1, 2022. And after the invasion from February 23rd, 2022 to September 24th, 2024.\n\n Click the button to render a slice of conflict events at the start of the war.")
            with gr.Column(): 
                filter_button = gr.Button("Render stories")
        
        story_view = gr.Plot(label="Story view")
        filter_button.click(show_story_view, outputs=[story_view])

        with gr.Row(): 
            with gr.Column(): 
                gr.Markdown(value="The underlying dataset supports predictions during the relative quiescent period from September 1st, 2020 to February 1, 2022. And after the invasion from February 23rd, 2022 to September 24th, 2024.\n\n Click the button to render a slice of conflict events at the start of the war.")
            with gr.Column(): 
                predict_button = gr.Button("Predict events")
        
        story_prediction = gr.Plot(label="Predictions")
        predict_button.click(show_story_predictions, outputs=[story_prediction])
        
        # One-shot load operations
        demo.load(render_map, outputs=[basemap])
        demo.load(render_points, outputs=[fud])
        demo.load(render_story,outputs=[story])
        demo.load(render_points_w_story, outputs=[story_context])

    demo.launch(share=share)

if __name__ == "__main__":
    demo()