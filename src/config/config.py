# General Settings
SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"

# Collaborative Filtering Settings
K_NEIGHBORS = 20
SIMILARITY_METRIC = "adjusted_cosine"   # Options: 'cosine', 'pearson', 'adjusted_cosine'
MIN_RATINGS_PER_USER = 5
TEST_RATIO = 0.2

# Evaluation Settings
RELEVANCE_THRESHOLD = 3.5  # Used to convert ratings to binary relevance for NDCG/Precision/Recall
TOP_K_EVAL = 10            # For @k metrics (NDCG@k, Precision@k, Recall@k)

# Scalability Settings
USER_SAMPLE_FRACS = [0.1, 0.3, 0.5, 1.0]

# Visualization Settings
PLOT_METRICS = ["RMSE", "MAE", "NDCG@k"]
