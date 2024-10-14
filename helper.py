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
    ranked_items = metadf.nlargest(10, 'bayesian_average')
    
    return ranked_items.sort_values('bayesian_average', ascending=False).head(10)

def most_popular_products_by_category(metadf, category, top_n=10):
    
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
    
def content_based_recommendations(metadf, item_title, top_n=10, chunk_size=1000):
    # Check if the item title exists in the metadata
    if item_title not in metadf['title'].values:
        print(f"Item '{item_title}' not found in the dataset.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for the 'description' column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to the descriptions
    tfidf_matrix = tfidf_vectorizer.fit_transform(metadf['description'].fillna(''))

    # Get the index of the item based on its title
    item_index = metadf[metadf['title'] == item_title].index[0]

    # Initialize an empty list to store similarities
    all_similar_items = []

    # Process the similarity matrix in chunks to avoid memory issues
    n_items = tfidf_matrix.shape[0]
    
    for start_idx in range(0, n_items, chunk_size):
        end_idx = min(start_idx + chunk_size, n_items)
        cosine_sim_chunk = cosine_similarity(tfidf_matrix[start_idx:end_idx], tfidf_matrix[item_index])
        
        # Store the similarity values and their respective indices
        chunk_similar_items = [(i + start_idx, sim[0]) for i, sim in enumerate(cosine_sim_chunk)]
        all_similar_items.extend(chunk_similar_items)

    # Sort similar items by similarity score (descending order)
    all_similar_items = sorted(all_similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = [item for item in all_similar_items if item[0] != item_index][:top_n]

    # Extract the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the recommended items
    recommended_items_details = metadf.iloc[recommended_item_indices][['title', 'average_rating', 'rating_number', 'images', 'price', 'store']]

    return recommended_items_details