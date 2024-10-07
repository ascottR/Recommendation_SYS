import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def most_popular_products(metadf):

    # Calculate the overall average rating
    overall_average = metadf['average_rating'].mean()

    # Set a minimum count of ratings to consider (75th percentile)
    C = metadf['rating_number'].quantile(0.75)

    # Calculate Bayesian average for each item
    metadf['bayesian_average'] = (metadf['rating_number'] * metadf['average_rating'] + C * overall_average) / (metadf['rating_number'] + C)

    # Rank items based on the Bayesian average
    ranked_items = metadf.nlargest(10, 'bayesian_average')
    
    return ranked_items.sort_values('bayesian_average', ascending=False).head(10)

def create_X_custom(df):
    """
    Generates a sparse matrix from a DataFrame containing user ratings for products.
    """
    # Ensure the rating column contains numeric values
    df['rating'] = df['rating'].astype(float)

    # Count unique users and items
    M = df['user_id'].nunique()  # Unique users
    N = df['parent_asin'].nunique()  # Unique items

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    item_mapper = dict(zip(np.unique(df["parent_asin"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    item_inv_mapper = dict(zip(list(range(N)), np.unique(df["parent_asin"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [item_mapper[i] for i in df['parent_asin']]

    # Create the sparse matrix
    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper

def find_similar_products_by_title(product_title, X, item_mapper, item_inv_mapper, product_titles, k, metric='cosine'):
    """
    Finds k-nearest neighbours for a given product title.
    """
    product_id = None
    for pid, title in product_titles.items():
        if title.lower() == product_title.lower():
            product_id = pid
            break

    if product_id is None:
        return []  # Return an empty list if the product is not found

    X = X.T
    neighbour_ids = []
    product_ind = item_mapper[product_id]
    product_vec = X[product_ind].reshape(1, -1)

    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    kNN.fit(X)

    neighbour = kNN.kneighbors(product_vec, return_distance=False)

    for i in range(1, k + 1):
        n = neighbour.item(i)
        neighbour_ids.append(item_inv_mapper[n])

    return neighbour_ids