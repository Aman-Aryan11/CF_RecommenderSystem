import time
import numpy as np
from src.evaluation.metrics import mae_rmse

def timed_evaluate_model(name, rating_matrix, similarity_func, test_df, model_type="user", k=20):
    """
    Time and evaluate a user/item-based CF model.

    Parameters:
        name (str): Model name for display
        rating_matrix (torch.Tensor): Full user-item rating matrix
        similarity_func (function): Function that returns a similarity matrix
        test_df (pd.DataFrame): Test data with user_idx and item_idx
        model_type (str): "user" or "item"
        k (int): Number of neighbors

    Returns:
        dict: Metrics + runtime + throughput
    """
    import torch
    from src.model.user_cf import predict_user_based
    from src.model.item_cf import predict_item_based

    # Step 1: Compute similarity matrix
    start_time = time.time()
    sim_matrix = similarity_func(rating_matrix if model_type == "user" else rating_matrix.T)
    similarity_time = time.time() - start_time

    # Step 2: Predict for test set
    y_true = []
    y_pred = []

    start_pred_time = time.time()
    for row in test_df.itertuples():
        user_id, item_id = row.user_idx, row.item_idx
        true_rating = row.Rating

        if model_type == "user":
            pred = predict_user_based(user_id, item_id, rating_matrix, sim_matrix, k)
        else:
            pred = predict_item_based(user_id, item_id, rating_matrix, sim_matrix, k)

        y_true.append(true_rating)
        y_pred.append(pred)

    pred_time = time.time() - start_pred_time
    total_time = similarity_time + pred_time
    throughput = len(y_true) / pred_time if pred_time > 0 else 0

    # Step 3: Metrics
    mae, rmse = mae_rmse(y_true, y_pred)

    return {
        "Model": name,
        "K": k,
        "MAE": mae,
        "RMSE": rmse,
        "Similarity Time (s)": round(similarity_time, 3),
        "Prediction Time (s)": round(pred_time, 3),
        "Total Time (s)": round(total_time, 3),
        "Throughput(preds/sec)": round(throughput, 2),
    }
