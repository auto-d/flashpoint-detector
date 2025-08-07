import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

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
    
def score (dataset, preds, test_ixs, threshold=0.3): 
    """
    Generic scoring method for our predictions     
    """
    
    score_ixs = {
        'precision': 0, 
        'recall' : 1, 
        'f1' : 2,
        'accuracy': 3
    }

    print(f"Scoring predictions... ")

    scores = np.zeros((len(preds), 4), dtype=np.float32)
        
    # Iterate over all provided stories. Grab the ground truth and check the predictions
    # Note we don't penalize (or reward) for cells which we have no ground truth for... 
    for i, ix in enumerate(test_ixs):
                
        labels = dataset.flatten_labels([ix])[0]        
        label_ixs = np.nonzero(labels)
        
        y = labels[label_ixs].copy()
        y[y == -1] = 0

        y_hat = preds[i][label_ixs]
                
        y_hat[y_hat >= threshold] = 1
        y_hat[y_hat < threshold] = 0

        scores[i][score_ixs['precision']] = precision_score(y, y_hat)
        scores[i][score_ixs['recall']] = recall_score(y, y_hat)
        scores[i][score_ixs['f1']] = f1_score(y, y_hat)
        scores[i][score_ixs['accuracy']] = accuracy_score(y, y_hat)

        print(y, y_hat)        
        print(f"Story {i} scores = {scores[i]}")
        
    return scores