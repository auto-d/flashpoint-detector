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

gdf = None
admin = None 

def initialize(tag):         
    """
    Initialize global state and populate initial recommendation s
    """

    global gdf 
    global admin 
    
    dataset = FlashpointsDataset.load("data", tag=tag)
    gdf = dataset.load_fud()    

    ukraine_admin_gdb = "data/ukr_admbnd_sspe_20240416_AB_GDB.gdb"    
    admin = gpd.read_file(ukraine_admin_gdb, layer=7)
    
    # Recover our model and create an empty dataset for our user(s)
    #model = nn.load_model("models/") 

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
    
def render_story(): 
    global gdf

    return render_box(make_box(2,3,0,3,5))

def render_points(steps=1): 
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

    fig, ax = plt.subplots(figsize=(8, 8))
    admin.plot(ax=ax)
    ax.axis('off')
    
    return fig 

def update_timestep(step): 

    return render_points(step)

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

        #FOlium
        # fmap = Map(
        #     location=[48, 31],
        #     zoom_start=6, 
        #     #crs="EPSG4326"
        #     )
        # map = Folium(value=fmap, height=800)
        #data = gr.DataFrame(value=gdf, height=200)
        #data.select(select, data, map)

        # Basemap 
        gr.Markdown("# 🗺️ Study Area")
        with gr.Row():
            with gr.Column(scale=2): 
                gr.Markdown(value="The study area is the entirety of the internationally recognized borders of Ukraine")
            with gr.Column(scale=1):
                basemap = gr.Plot(label="Basemap", container=False,)

        # FUD points
        gr.Markdown("# 📍 Flashpoints Events")
        with gr.Row():                             
            with gr.Column(scale=1): 
                gr.Markdown(value="The Flashpoints Ukraine Dataset (FUD) labels NASA thermal anomalies where their geometry intersects with ACLED-reported conflict events. We elect to operate on a granularity of 1 or more days, depending on dataset build settings. Higher resolutions in the temporal dimension create issues given satellite overflight patterns and the maximum resolution available in our event reporting.")
            with gr.Column(scale=2): 
                time_slider = gr.Slider(label="Thermal events to render", minimum=1, value=1, maximum=10, step=1)                    
                fud = gr.Plot(label="Flashpoints Ukraine Dataset Slices") 
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
                story_context = gr.Plot(label="Story context")
            with gr.Column(scale=1): 
                gr.Markdown(value="The geomtry of a single story is rendered to the left.")                

        demo.load(render_map, outputs=[basemap])
        demo.load(render_points, outputs=[fud])
        demo.load(render_story,outputs=[story])
        demo.load(render_points_w_story, outputs=[story_context])

        #     gr.Markdown(value="*If you'd like to see more recommendations, adjust accordingly with the slider to the right.*\n\n**Note:** Our low-quality items often have fragmented listings (which could be part of why they are so terrible!). When we encounter these we filter from the listing. *Actual recommendation counts may not match recommendation slider accordingly!* ☺️")            
        
        
        # gallery = gr.Gallery(label="AI Recommendations", columns=3, height="auto", 
        #                      value = get_product_images())         
        
        # with gr.Column(): 
        #     product_info = gr.Markdown(label="Product Information", value="Select an item to display more product information!")        
        #     rating = gr.Radio(choices=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], label="Rate this product!")
        #     rating.change(fn=submit_rating, inputs=rating, outputs=[gallery, rating])

        #     gallery.select(fn=on_click, outputs=product_info)
        #     

    demo.launch(share=share)

if __name__ == "__main__":
    demo()