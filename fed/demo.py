import gradio as gr
import pandas as pd
import numpy as np
from fed.dataset import FlashpointsDataset, FlashpointsTorchDataset
from . import nn
from . import similarity
from gradio_folium import Folium
from folium import Map

from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

gdf = None

def initialize(tag):         
    """
    Initialize global state and populate initial recommendation s
    """

    global gdf 
    
    dataset = FlashpointsDataset.load(data_dir="../data", tag=tag)
    gdf = dataset.load_fud()    

    # Recover our model and create an empty dataset for our user(s)
    model = nn.load_model("models/") 

def select(df, data: gr.SelectData):
    row = df.iloc[data.index[0], :]
    return Map(location=[row['Latitude'], row['Longitude']])

def demo(share=False, data_tag="test"): 
    """
    Our Gradio demo app!

    NOTE: with snippets from folium, gradio and gradio_folium documentation 
    """
    global gdf

    initialize(data_tag)         

    demo = gr.Blocks()
    with demo: 

        # Header         
        gr.Markdown(value="# 💥 Flashpoint Event Detector")
        gr.Markdown(value="##  Ukraine conflict event detection through NASA thermal anomaly data analysis.")
        gr.Markdown(value="The flashpoint ukraine detector leverages four years of thermal anomaly data from multiple NASA hyperspectral imagers contextualized by the Armed Conflict & Location Data (ACLED) source. Its purpose is to classify new thermal anomalies to disambiguate conflict events from other thermal sources such as naturally occuring forest fires.")

        gr.Markdown(value="Below you can explore the historical dataset and run predictions.")

        map = Folium(value=Map(location=[23, 45]), height=400)
        data = gr.DataFrame(value=gdf, height=200)
        data.select(select, data, map)

        # with gr.Row():             
        #     gr.Markdown(value="*If you'd like to see more recommendations, adjust accordingly with the slider to the right.*\n\n**Note:** Our low-quality items often have fragmented listings (which could be part of why they are so terrible!). When we encounter these we filter from the listing. *Actual recommendation counts may not match recommendation slider accordingly!* ☺️")            
        #     topk_slider = gr.Slider(label="Recommendations to generate", value=10, maximum=100, step=5)
        
        # gallery = gr.Gallery(label="AI Recommendations", columns=3, height="auto", 
        #                      value = get_product_images())         
        
        # with gr.Column(): 
        #     product_info = gr.Markdown(label="Product Information", value="Select an item to display more product information!")        
        #     rating = gr.Radio(choices=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], label="Rate this product!")
        #     rating.change(fn=submit_rating, inputs=rating, outputs=[gallery, rating])

        #     gallery.select(fn=on_click, outputs=product_info)
        #     topk_slider.change(fn=update_topk, inputs=topk_slider, outputs=gallery)

    demo.launch(share=share)

if __name__ == "__main__":
    demo()