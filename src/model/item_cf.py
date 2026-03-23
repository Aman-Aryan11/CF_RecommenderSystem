import torch
import torch.nn.functional as F

def compute_item_similarity(matrix, similarity="cosine"):
    """
    Compute item-item similarity matrix.

    Parameters:
        matrix (torch.Tensor): Item-user matrix [num_items, num_users]
        similarity (str): 'cosine', 'adjusted_cosine', or 'pearson'

    Returns:
        torch.Tensor: Similarity matrix [num_items, num_items]
    """
    if similarity == "cosine":
        norm_matrix = F.normalize(matrix, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    elif similarity == "adjusted_cosine":
        # Subtract user mean from each column (i.e., across items)
        mask = matrix != 0
        num_items, num_users = matrix.shape

        # Compute user means ignoring zeros
        user_sums = matrix.sum(dim=0)
        user_counts = mask.sum(dim=0).clamp(min=1)  # avoid division by 0
        user_means = user_sums / user_counts        # shape: [num_users]

        # Broadcast and center
        matrix_centered = matrix - user_means.unsqueeze(0)  # [num_items, num_users]
        matrix_centered = matrix_centered * mask            # zero-out unrated

        norm_matrix = F.normalize(matrix_centered, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    elif similarity == "pearson":
        # Subtract item mean from each row (i.e., across users)
        mask = matrix != 0
        num_items, num_users = matrix.shape

        item_sums = matrix.sum(dim=1)
        item_counts = mask.sum(dim=1).clamp(min=1)
        item_means = item_sums / item_counts       # shape: [num_items]

        # Broadcast and center
        matrix_centered = matrix - item_means.unsqueeze(1)  # [num_items, num_users]
        matrix_centered = matrix_centered * mask

        norm_matrix = F.normalize(matrix_centered, p=2, dim=1)
        return torch.mm(norm_matrix, norm_matrix.T)

    else:
        raise ValueError(f"Unsupported similarity metric: {similarity}")


def predict_item_based(user_id, item_id, rating_matrix, similarity_matrix, k=20):
    """
    Predict a rating for a user-item pair using item-based collaborative filtering.

    Parameters:
        user_id (int): Index of the user (user_idx)
        item_id (int): Index of the item (item_idx)
        rating_matrix (torch.Tensor): User-Item matrix [num_users, num_items]
        similarity_matrix (torch.Tensor): Item-Item similarity matrix
        k (int): Number of nearest neighbors to use

    Returns:
        predicted_rating (float)
    """
    # Ratings given by this user
    user_ratings = rating_matrix[user_id]

    # Similarities of target item with all other items
    item_sim = similarity_matrix[item_id]

    # Mask items that the user hasn't rated
    mask = user_ratings > 0
    similarities = item_sim[mask]
    ratings = user_ratings[mask]

    # Get top-k similar items
    if len(similarities) == 0:
        return 0.0  # can't predict if no neighbors

    topk = min(k, len(similarities))
    topk_indices = torch.topk(similarities, topk).indices

    topk_sims = similarities[topk_indices]
    topk_ratings = ratings[topk_indices]

    # Weighted average
    numerator = torch.sum(topk_sims * topk_ratings)
    denominator = torch.sum(torch.abs(topk_sims))

    if denominator == 0:
        return 0.0
    else:
        return (numerator / denominator).item()
