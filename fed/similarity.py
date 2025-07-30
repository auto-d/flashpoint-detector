import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(a, b): 
    """
    Return pairwise similarity between elements in passed arrays
    """
    similarity_matrix = cosine_similarity(a,b)
    pairwise_similarity = np.diag(similarity_matrix)[0]

    return float(pairwise_similarity)

def pearson_similarity(a, b):
    """
    Compute Pearson similarity
    """
    return (1 + pearsonr(a, b).statistic) / 2
