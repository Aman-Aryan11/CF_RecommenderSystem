import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mae_rmse(y_true, y_pred):
    """
    Compute MAE and RMSE between true and predicted ratings.

    Parameters:
        y_true (list or np.array)
        y_pred (list or np.array)

    Returns:
        (mae, rmse): Tuple of floats
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def ndcg_at_k(y_true, y_scores, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Parameters:
        y_true (list of int): Binary relevance (1 for relevant, 0 for not)
        y_scores (list of float): Predicted scores (e.g., predicted ratings)
        k (int)

    Returns:
        NDCG@k (float)
    """
    order = np.argsort(y_scores)[::-1]
    y_true = np.take(y_true, order[:k])

    dcg = np.sum((2 ** y_true - 1) / np.log2(np.arange(2, len(y_true) + 2)))
    ideal = np.sum((2 ** sorted(y_true, reverse=True) - 1) / np.log2(np.arange(2, len(y_true) + 2)))

    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(y_true, y_scores, k=10):
    """
    Compute Precision@K: fraction of top-K items that are relevant.

    Parameters:
        y_true (list of int): Binary relevance
        y_scores (list of float): Predicted scores

    Returns:
        Precision@k (float)
    """
    order = np.argsort(y_scores)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / k


def recall_at_k(y_true, y_scores, k=10):
    """
    Compute Recall@K: fraction of relevant items that are in top-K.

    Parameters:
        y_true (list of int): Binary relevance
        y_scores (list of float): Predicted scores

    Returns:
        Recall@k (float)
    """
    order = np.argsort(y_scores)[::-1]
    y_true = np.asarray(y_true)
    hits = np.take(y_true, order[:k])
    total_relevant = np.sum(y_true)
    return np.sum(hits) / total_relevant if total_relevant > 0 else 0.0
