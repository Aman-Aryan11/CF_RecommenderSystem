import numpy as np
import pandas as pd

def user_based_split(df, test_ratio=0.2, min_ratings=5, seed=42):
    """
    Perform a user-aware train-test split:
    - Ensures each user appears in the training set
    - Leaves all ratings in train if user has < min_ratings

    Parameters:
        df (pd.DataFrame): Ratings DataFrame with 'user_idx' and 'item_idx'
        test_ratio (float): Fraction of ratings to place in test set per user
        min_ratings (int): Minimum ratings a user must have to be split
        seed (int): Random seed for reproducibility

    Returns:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
    """
    np.random.seed(seed)

    train_data = []
    test_data = []

    grouped = df.groupby('user_idx')

    for user_id, group in grouped:
        n_items = len(group)
        if n_items < min_ratings:
            train_data.extend(group.to_dict('records'))
        else:
            test_size = int(n_items * test_ratio)
            test_indices = np.random.choice(group.index, size=test_size, replace=False)
            train_indices = list(set(group.index) - set(test_indices))

            train_data.extend(df.loc[train_indices].to_dict('records'))
            test_data.extend(df.loc[test_indices].to_dict('records'))

    return pd.DataFrame(train_data), pd.DataFrame(test_data)
