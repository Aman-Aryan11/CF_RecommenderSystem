import os
import pandas as pd

def load_data(data_dir="data"):
    """
    Load ratings, users, and movies datasets from the given directory.

    Parameters:
        data_dir (str): Path to the folder containing CSV files.

    Returns:
        ratings_df (pd.DataFrame): DataFrame with ratings data.
        users_df (pd.DataFrame): DataFrame with user demographic data.
        movies_df (pd.DataFrame): DataFrame with movie metadata.
    """

    ratings_path = os.path.join(data_dir, "ratings.csv")
    users_path   = os.path.join(data_dir, "users.csv")
    movies_path  = os.path.join(data_dir, "movies.csv")

    try:
        ratings_df = pd.read_csv(ratings_path)
        users_df   = pd.read_csv(users_path)
        movies_df  = pd.read_csv(movies_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load data file: {e.filename}")

    return ratings_df, users_df, movies_df
