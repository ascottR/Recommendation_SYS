import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def most_popular_products(metadf):

    # Calculate the overall average rating
    overall_average = metadf['average_rating'].mean()

    # Set a minimum count of ratings to consider (75th percentile)
    C = metadf['rating_number'].quantile(0.75)

    # Calculate Bayesian average for each item
    metadf['bayesian_average'] = (metadf['rating_number'] * metadf['average_rating'] + C * overall_average) / (metadf['rating_number'] + C)

    # Rank items based on the Bayesian average
    ranked_items = metadf.nlargest(12, 'bayesian_average')
    
    return ranked_items.sort_values('bayesian_average', ascending=False).head(12)

def most_popular_products_by_category(metadf, category, top_n=12):
    
    # Filter the dataset by the selected category
    filtered_df = metadf[metadf['categories'].str.contains(category, case=False, na=False)]

    if filtered_df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no products are found

    # Calculate the overall average rating within the category
    overall_average = filtered_df['average_rating'].mean()

    # Set a minimum count of ratings to consider (75th percentile within the category)
    C = filtered_df['rating_number'].quantile(0.75)

    # Calculate Bayesian average for each item within the category
    filtered_df['bayesian_average'] = (
        (filtered_df['rating_number'] * filtered_df['average_rating'] + C * overall_average) /
        (filtered_df['rating_number'] + C)
    )

    # Rank items based on the Bayesian average
    ranked_items = filtered_df.nlargest(top_n, 'bayesian_average')
    
    return ranked_items.sort_values('bayesian_average', ascending=False)

# Function to get product by asin
def get_product_by_asin(parent_asin, metadf):
    # Query the DataFrame for the specific product
    product = metadf[metadf['parent_asin'] == parent_asin].to_dict(orient='records')
    # Return the first match if available
    return product[0] if product else None

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

def find_similar_products_by_id(product_id, X, item_mapper, item_inv_mapper, k, metric='cosine'):
    """
    Finds k-nearest neighbours for a given product id.

    Args:
        product_id: id of the product of interest
        X: user-item utility matrix (sparse matrix)
        k: number of similar products to retrieve
        metric: distance metric for kNN calculations

    Output: returns list of k similar product IDs
    """
    # Transpose the user-item matrix so products are the rows
    X = X.T
    neighbour_ids = []

    # Get the index of the product
    product_ind = item_mapper[product_id]
    product_vec = X[product_ind]

    # Reshape the product vector to be compatible with kneighbors
    if isinstance(product_vec, np.ndarray):
        product_vec = product_vec.reshape(1, -1)

    # Use k+1 since kNN output includes the product ID of interest
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    kNN.fit(X)

    # Find the nearest neighbours
    neighbour = kNN.kneighbors(product_vec, return_distance=False)

    # Collect similar product IDs, skipping the first one (the product itself)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(item_inv_mapper[n])

    return neighbour_ids
    
def content_based_recommendations(metadf, query, top_n=8, chunk_size=1000):
    # Filter products with titles that contain the query (case-insensitive)
    matching_products = metadf[metadf['title'].str.contains(query, case=False)]

    if matching_products.empty:
        print(f"No matching products found for query: '{query}'")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for the 'description' column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to the descriptions
    tfidf_matrix = tfidf_vectorizer.fit_transform(matching_products['description'].fillna(''))

    # Compute cosine similarity between all products and the first matching product
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()

    # Get the top N most similar items (excluding the query item itself)
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]

    # Get the details of the recommended items
    recommended_items = matching_products.iloc[similar_indices][
        ['title', 'average_rating', 'rating_number', 'images', 'price', 'store', 'parent_asin']
    ]

    return recommended_items
