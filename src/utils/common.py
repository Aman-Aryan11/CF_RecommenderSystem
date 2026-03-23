import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility across NumPy and PyTorch.

    Parameters:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rating_to_relevance(ratings, threshold=3.5):
    """
    Convert ratings to binary relevance (1 = relevant, 0 = not relevant).

    Parameters:
        ratings (list or np.array): Original numerical ratings
        threshold (float): Minimum score to consider an item relevant

    Returns:
        relevance_labels (np.array): Binary 0/1 labels
    """
    return np.array([1 if r >= threshold else 0 for r in ratings])


def describe_dataframe(df, name="DataFrame"):
    """
    Print basic info and shape of a DataFrame.

    Parameters:
        df (pd.DataFrame)
        name (str): Optional label
    """
    print(f"--- {name} ---")
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.head(), "\n")
