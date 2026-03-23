import pandas as pd
import numpy as np
import torch
from src.evaluation.runtime import timed_evaluate_model
from src.preprocessing.preprocess import build_rating_matrix
from sklearn.preprocessing import LabelEncoder

def scalability_analysis(
    rating_df,
    train_df,
    test_df,
    model_type="user",
    similarity="cosine",
    k=20,
    user_sample_fracs=[0.1, 0.3, 0.5, 1.0],
    seed=42
):
    """
    Measure how model scales with increasing training data size.

    Parameters:
        rating_df (pd.DataFrame): Full ratings with 'UserID', 'MovieID', 'Rating'
        train_df (pd.DataFrame): Full training set
        test_df (pd.DataFrame): Full test set
        model_type (str): "user" or "item"
        similarity (str): similarity method
        k (int): neighborhood size
        user_sample_fracs (list): user fractions to sample
        seed (int): random seed

    Returns:
        pd.DataFrame: results including runtime, throughput, and metrics
    """
    from src.model.user_cf import compute_user_similarity
    from src.model.item_cf import compute_item_similarity

    np.random.seed(seed)

    results = []

    for frac in user_sample_fracs:
        sampled_users = train_df['user_idx'].drop_duplicates().sample(frac=frac, random_state=seed)
        subset_train = train_df[train_df['user_idx'].isin(sampled_users)]
        subset_test = test_df[test_df['user_idx'].isin(sampled_users)]

        # Re-encode user/item IDs
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        subset_train = subset_train.copy()
        subset_train['user_idx'] = user_encoder.fit_transform(subset_train['user_idx'])
        subset_train['item_idx'] = item_encoder.fit_transform(subset_train['item_idx'])

        subset_test = subset_test[subset_test['user_idx'].isin(user_encoder.classes_)]
        subset_test = subset_test[subset_test['item_idx'].isin(item_encoder.classes_)]
        subset_test = subset_test.copy()
        subset_test['user_idx'] = user_encoder.transform(subset_test['user_idx'])
        subset_test['item_idx'] = item_encoder.transform(subset_test['item_idx'])

        num_users = subset_train['user_idx'].nunique()
        num_items = subset_train['item_idx'].nunique()

        rating_matrix_sub = build_rating_matrix(subset_train, num_users, num_items)

        similarity_func = (
            lambda mat: compute_user_similarity(mat, similarity=similarity)
            if model_type == "user"
            else lambda mat: compute_item_similarity(mat, similarity=similarity)
        )(rating_matrix_sub)

        # Evaluate and time
        result = timed_evaluate_model(
            name=f"{model_type.title()}-based ({similarity}) [{int(frac*100)}% users]",
            rating_matrix=rating_matrix_sub,
            similarity_func=lambda mat: similarity_func,
            test_df=subset_test,
            model_type=model_type,
            k=k
        )

        result["DataFraction"] = frac
        result["ModelType"] = model_type
        result["Similarity"] = similarity
        results.append(result)

    return pd.DataFrame(results)
