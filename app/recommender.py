import torch
from src.model.user_cf import compute_user_similarity, predict_user_based
from src.model.item_cf import compute_item_similarity, predict_item_based
from src.preprocessing.preprocess import build_rating_matrix

def generate_recommendations(
    ratings_df,
    user_encoder,
    item_encoder,
    user_id,
    model_type="user",
    similarity="cosine",
    k_neighbors=20,
    top_k=10
):
    """
    Generate top-k recommendations for a given user.

    Parameters:
        ratings_df (DataFrame): Ratings data with 'user_idx' and 'item_idx'
        user_encoder (LabelEncoder): Encoder for UserID → user_idx
        item_encoder (LabelEncoder): Encoder for MovieID → item_idx
        user_id (int): Actual user ID from dataset
        model_type (str): "user" or "item"
        similarity (str): Similarity metric: "cosine", "pearson", "adjusted_cosine"
        k_neighbors (int): Neighborhood size
        top_k (int): Number of recommendations to return

    Returns:
        List of tuples: (movie_id, predicted_score)
    """
    # Encode user index
    user_idx = user_encoder.transform([user_id])[0]
    num_users = ratings_df['user_idx'].nunique()
    num_items = ratings_df['item_idx'].nunique()

    # Build rating matrix
    rating_matrix = build_rating_matrix(ratings_df, num_users, num_items)
    item_user_matrix = rating_matrix.T

    # Compute similarity
    if model_type == "user":
        sim_matrix = compute_user_similarity(rating_matrix, similarity)
    elif model_type == "item":
        sim_matrix = compute_item_similarity(item_user_matrix, similarity)
    else:
        raise ValueError("Invalid model_type: choose 'user' or 'item'")

    # Filter items user hasn't rated
    user_rated_items = ratings_df[ratings_df['user_idx'] == user_idx]['item_idx'].tolist()
    predictions = []

    for item_idx in range(num_items):
        if item_idx in user_rated_items:
            continue

        if model_type == "user":
            score = predict_user_based(user_idx, item_idx, rating_matrix, sim_matrix, k=k_neighbors)
        else:
            score = predict_item_based(user_idx, item_idx, rating_matrix, sim_matrix, k=k_neighbors)

        predictions.append((item_idx, score))

    # Top-k recommendations
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]

    # Decode item indices back to MovieIDs
    movie_ids = item_encoder.inverse_transform([idx for idx, _ in top_predictions])
    scores = [round(score, 2) for _, score in top_predictions]

    return list(zip(movie_ids, scores))
