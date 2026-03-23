import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.config.config import *
from src.data_ingestion.load_data import load_data
from src.preprocessing.preprocess import encode_user_item_ids
from src.utils.common import set_seed
from app.recommender import generate_recommendations

# Setup
set_seed(SEED)
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load and cache data
@st.cache_data
def load_all():
    ratings, users, movies = load_data(DATA_DIR)
    ratings, user_encoder, item_encoder = encode_user_item_ids(ratings)
    return ratings, users, movies, user_encoder, item_encoder

ratings_df, users_df, movies_df, user_encoder, item_encoder = load_all()

# Sidebar - settings
st.sidebar.title("⚙️ Recommender Settings")
model_type = st.sidebar.selectbox("Model Type", ["User-based", "Item-based"])
similarity = st.sidebar.selectbox("Similarity Metric", ["cosine", "pearson", "adjusted_cosine"])
k_neighbors = st.sidebar.slider("Neighborhood Size (K)", min_value=5, max_value=50, value=20)
top_k = st.sidebar.slider("Top-K Recommendations", min_value=5, max_value=20, value=10)

# Main UI
st.title("🎬 Movie Recommender System")
user_id_input = st.number_input("Enter User ID", min_value=int(ratings_df['UserID'].min()), max_value=int(ratings_df['UserID'].max()))

if st.button("Get Recommendations"):
    if user_id_input not in ratings_df['UserID'].values:
        st.warning("❌ User ID not found in dataset.")
    else:
        # Call recommendation logic
        recs = generate_recommendations(
            ratings_df=ratings_df,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            user_id=user_id_input,
            model_type=model_type.lower().split("-")[0],  # user or item
            similarity=similarity,
            k_neighbors=k_neighbors,
            top_k=top_k
        )

        if not recs:
            st.info("No recommendations available for this user.")
        else:
            movie_ids, scores = zip(*recs)
            titles = movies_df.set_index("MovieID").loc[list(movie_ids), "Title"].values

            result_df = pd.DataFrame({
                "Movie": titles,
                "Predicted Rating": [f"{score:.2f}" for score in scores]
            })

            st.subheader(f"📽️ Top {top_k} Recommendations for User {user_id_input}")
            st.table(result_df)
