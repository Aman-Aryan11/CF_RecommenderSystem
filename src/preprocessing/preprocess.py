import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def encode_user_item_ids(df):
    """
    Encode UserID and MovieID into 0-based indices: user_idx and item_idx.

    Parameters:
        df (pd.DataFrame): Original ratings DataFrame with 'UserID' and 'MovieID'.

    Returns:
        df (pd.DataFrame): Updated DataFrame with 'user_idx' and 'item_idx' columns.
        user_encoder (LabelEncoder): Fitted encoder for users.
        item_encoder (LabelEncoder): Fitted encoder for items.
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_idx'] = user_encoder.fit_transform(df['UserID'])
    df['item_idx'] = item_encoder.fit_transform(df['MovieID'])

    return df, user_encoder, item_encoder


def build_rating_matrix(df, num_users, num_items):
    """
    Build a dense rating matrix (torch tensor) from the encoded DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'user_idx', 'item_idx', 'Rating'
        num_users (int): Number of unique users
        num_items (int): Number of unique items

    Returns:
        torch.Tensor: Rating matrix of shape [num_users, num_items]
    """
    rating_matrix = torch.zeros((num_users, num_items))

    for row in df.itertuples():
        rating_matrix[row.user_idx, row.item_idx] = row.Rating

    return rating_matrix
