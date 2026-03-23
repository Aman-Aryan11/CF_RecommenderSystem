## Collaborative Filtering Movie Recommender

This project implements a movie recommendation system using memory-based collaborative filtering on the MovieLens 1M dataset. It includes:

- User-based collaborative filtering
- Item-based collaborative filtering
- Multiple similarity functions: `cosine`, `pearson`, and `adjusted_cosine`
- Offline evaluation scripts for rating prediction and ranking quality
- A local Streamlit app for interactive recommendations

The codebase is organized so you can experiment from the command line and also serve a simple web UI locally.

## Dataset

The repository contains MovieLens data in the `data/` directory:

- `ratings.csv` / `ratings.dat`
- `movies.csv` / `movies.dat`
- `users.csv` / `users.dat`

The project reads the CSV files by default. These files contain user ratings, movie metadata, and user demographic information derived from the MovieLens 1M dataset.

## Features

- Dense user-item rating matrix construction with PyTorch
- User-user and item-item similarity computation
- Top-K neighborhood-based rating prediction
- Metrics for regression and ranking:
  - `MAE`
  - `RMSE`
  - `NDCG@K`
  - `Precision@K`
  - `Recall@K`
- User-aware train/test split
- Interactive recommendation app built with Streamlit

## Project Structure

```text
CF/
├── app/
│   ├── main.py                # Streamlit interface
│   └── recommender.py         # Recommendation generation for the UI
├── data/                      # MovieLens CSV and DAT files
├── experiments/
│   ├── run_all.py             # Benchmark user/item CF across settings
│   ├── run_user_cf.py         # Evaluate user-based CF
│   └── run_item_cf.py         # Evaluate item-based CF
├── notebook/                  # EDA and model notebooks
├── results/                   # Saved experiment outputs
├── src/
│   ├── config/config.py       # Central configuration
│   ├── data_ingestion/        # Dataset loading
│   ├── preprocessing/         # Encoding and matrix building
│   ├── model/                 # User/item CF implementations
│   ├── evaluation/            # Metrics and runtime utilities
│   └── utils/                 # Common helpers, split logic, plotting
├── ResearchPaper/             # Supporting research material
├── requirements.txt
└── README.md
```

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

If you want one explicit install command, you can also use:

```bash
pip install pandas numpy scipy torch matplotlib seaborn scikit-learn tqdm streamlit
```

## Configuration

Project-wide settings live in `src/config/config.py`.

Important defaults:

- `DATA_DIR = "data"`
- `RESULTS_DIR = "results"`
- `K_NEIGHBORS = 20`
- `SIMILARITY_METRIC = "adjusted_cosine"`
- `TEST_RATIO = 0.2`
- `TOP_K_EVAL = 10`

You can update these values to change the default experiment behavior.

## How It Works

### 1. Data loading

`src/data_ingestion/load_data.py` loads:

- `ratings.csv`
- `users.csv`
- `movies.csv`

### 2. Preprocessing

`src/preprocessing/preprocess.py`:

- encodes `UserID` to `user_idx`
- encodes `MovieID` to `item_idx`
- builds a dense user-item rating matrix

### 3. Modeling

Two recommenders are implemented:

- `src/model/user_cf.py`
- `src/model/item_cf.py`

Each supports:

- `cosine`
- `pearson`
- `adjusted_cosine`

Predictions are made with a weighted average over the top `k` most similar neighbors.

### 4. Evaluation

`src/evaluation/metrics.py` provides:

- `mae_rmse`
- `ndcg_at_k`
- `precision_at_k`
- `recall_at_k`

`src/utils/split.py` performs a user-aware train/test split so each user can remain represented in training.

## Running Experiments

### Run user-based collaborative filtering

```bash
python -m experiments.run_user_cf
```

### Run item-based collaborative filtering

```bash
python -m experiments.run_item_cf
```

### Run the full comparison grid

```bash
python -m experiments.run_all
```

This script evaluates:

- both user-based and item-based CF
- three similarity functions
- multiple neighborhood sizes: `10`, `20`, and `40`

The combined results are saved to:

```text
results/all_results.csv
```

## Running the Streamlit App

Start the local app with:

```bash
streamlit run app/main.py
```

If you want it pinned to a specific local port:

```bash
streamlit run app/main.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

### App workflow

In the sidebar, choose:

- model type: `User-based` or `Item-based`
- similarity metric
- neighborhood size `K`
- number of recommendations to return

Then enter a valid `User ID` from the dataset and click `Get Recommendations`.

The app returns the top recommended movie titles with predicted ratings.

## Example Development Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit
python -m experiments.run_all
streamlit run app/main.py
```

## Notes and Limitations

- The recommendation pipeline builds a dense rating matrix, which is simple and clear but less memory-efficient than a sparse implementation.
- Predictions default to `0.0` when there are no usable neighbors or the similarity denominator is zero.
- The app computes recommendations on demand, so larger experiments are better run through the scripts in `experiments/`.
- `requirements.txt` includes core scientific dependencies, but `streamlit` is required separately for the web UI in the current repository state.

## References

- MovieLens dataset: GroupLens Research, University of Minnesota
- Research material included in `ResearchPaper/`

## License and Data Usage

This repository contains MovieLens-derived data files. Please follow the original MovieLens usage and citation requirements when using the dataset in research or redistributed work.
