import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI(api_key="none", base_url="http://localhost:11434/v1")

def embed(text): 
    """
    Generate a text embedding for the input sequence using our trusty ollama 
    nomin embedding model 

    NOTE: snippet from https://platform.openai.com/docs/guides/embeddings
    """    
    response = client.embeddings.create(model="nomic-embed-text:latest", input=text)
    return response.data[0].embedding

def cosine_similarity(a, b): 
    """
    Return pairwise similarity between elements in passed arrays
    """
    similarity_matrix = cosine_similarity([embed(a)],[embed(b)])
    pairwise_similarity = np.diag(similarity_matrix)[0]

    return float(pairwise_similarity)

def pearson_similarity(a, b):
    """
    Compute Pearson similarity
    """
    return (1 + pearsonr(a, b).statistic) / 2

def find_key(dict_, val): 
    """
    Return the key associated with a value
    """
    for k, v in dict_.items(): 
        if v == val: 
            return k

def find_keys(dict_, vals): 
    """
    Return the keys associated with a value list
    """
    keys = []
    for val in vals: 
        key = find_key(dict_, val)
        if key: 
            keys.append(key)
        else: 
            print(f"* find_keys(): WARNING - failed to identify key associated with value {val}, omitting from results!")
    
    return keys

def map_keys(dict_a, a_vals, dict_b): 
    """
    Map unique values in dict A to their corresponding values in 
    dict B by way of (hopefully) common keys. 
    """
    keys = find_keys(dict_a, a_vals)
    b_vals = []
    for k in keys: 
        if k in dict_b: 
            b_vals.append(dict_b[k])
        else: 
            print(f"* map_keys(): WARNING - failed to map key {k} between dicts, omitting from mapping!")
    
    return b_vals

def argmax(list_, exclude, include): 
    """
    Argmax with a list of indices to include, and exclude
    """    
    max_i = 0 
    for i, a in enumerate(list_): 
        if i not in exclude: 
            if include and i not in include: 
                continue 
            if a >= list_[max_i]: 
                max_i = i 
    return max_i 

def argmax2(list_, exclude, include): 
    max_i = 0
    for i in include: 
        if i not in exclude: 
            if list_[i] >= list_[max_i]: 
                max_i = i 
    return max_i 

def argmin(list_, exclude, include): 
    """
    Argmin with a list of indices to include, and exclude
    """    
    min_i = 0 
    for i, a in enumerate(list_): 
        if i not in exclude: 
            if include and i not in include: 
                continue
            if a <= list_[min_i]: 
                min_i = i 
    return min_i 