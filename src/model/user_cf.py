import torch
import torch.nn.functional as F

def compute_user_similarity(matrix, similarity="cosine"):
    """
    Compute user-user similarity matrix.

    Parameters:
        matrix (torch.Tensor): User-Item matrix [num_users, num_items]
        similarity (str): 'cosine', 'adjusted_cosine', or 'pearson'

    Returns:
        torch.Tensor: Similarity matrix [num_users, num_users]
    """
    if similarity == "cosine":
        norm_matrix = F.normalize(matrix, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    elif similarity == "adjusted_cosine":
        # Subtract item mean across users (adjusted cosine: normalize across items)
        mask = matrix != 0
        num_users, num_items = matrix.shape

        item_sums = matrix.sum(dim=0)
        item_counts = mask.sum(dim=0).clamp(min=1)
        item_means = item_sums / item_counts  # shape: [num_items]

        matrix_centered = matrix - item_means.unsqueeze(0)  # broadcast across rows
        matrix_centered = matrix_centered * mask  # zero-out unrated

        norm_matrix = F.normalize(matrix_centered, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    elif similarity == "pearson":
        # Subtract user mean (row-wise mean centering)
        mask = matrix != 0
        num_users, num_items = matrix.shape

        user_sums = matrix.sum(dim=1)
        user_counts = mask.sum(dim=1).clamp(min=1)
        user_means = user_sums / user_counts  # shape: [num_users]

        matrix_centered = matrix - user_means.unsqueeze(1)  # broadcast across columns
        matrix_centered = matrix_centered * mask

        norm_matrix = F.normalize(matrix_centered, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    else:
        raise ValueError(f"Unsupported similarity metric: {similarity}")


def predict_user_based(user_id, item_id, rating_matrix, similarity_matrix, k=20):
    """
    Predict a rating for a user-item pair using user-based collaborative filtering.

    Parameters:
        user_id (int): Index of the user (user_idx)
        item_id (int): Index of the item (item_idx)
        rating_matrix (torch.Tensor): User-Item matrix [num_users, num_items]
        similarity_matrix (torch.Tensor): User-User similarity matrix
        k (int): Number of nearest neighbors to use

    Returns:
        predicted_rating (float)
    """
    item_ratings = rating_matrix[:, item_id]  # Ratings for this item by all users
    user_sim = similarity_matrix[user_id]     # Similarities between this user and all others

    # Only consider users who have rated the item
    mask = item_ratings > 0
    relevant_sims = user_sim[mask]
    relevant_ratings = item_ratings[mask]

    if len(relevant_sims) == 0:
        return 0.0  # No neighbors rated this item

    topk = min(k, len(relevant_sims))
    topk_indices = torch.topk(relevant_sims, topk).indices

    topk_sims = relevant_sims[topk_indices]
    topk_ratings = relevant_ratings[topk_indices]

    numerator = torch.sum(topk_sims * topk_ratings)
    denominator = torch.sum(torch.abs(topk_sims))

    if denominator == 0:
        return 0.0
    else:
        return (numerator / denominator).item()
