from flask import Flask, render_template, request
import pandas as pd
import ast
from helper import most_popular_products, create_X_custom, find_similar_products_by_title
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load your dataset
metadf = pd.read_csv('artifacts/meta_final.csv')
reviewdf = pd.read_csv('artifacts/filtered_dataset.csv')

# Function to extract 'large' image from the 'Images' column
def extract_large_image(image_column):
    try:
        # Convert the string representation of the dictionary to an actual dictionary
        image_dict = ast.literal_eval(image_column)
        return image_dict.get('large', None)  
    except (ValueError, SyntaxError):
        return None

# Prepare the user-item matrix for collaborative filtering
X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper = create_X_custom(reviewdf)

@app.route('/')
def home():
    # Get the most popular products
    popular_products = most_popular_products(metadf)

    # Extract 'large' images for each product
    popular_products['large_image_url'] = popular_products['images'].apply(extract_large_image)

    # Convert the products DataFrame to a dictionary list for rendering
    popular_products_list = popular_products.to_dict(orient='records')

    return render_template('index.html', popular_products=popular_products_list)

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/recommendations', methods=['POST'])
def recommend():
    product_title = request.form['prod']  # Get the product title from the form
    k = int(request.form.get('nbr', 5))  # Default to 5 recommendations if not specified

    similar_product_ids = find_similar_products_by_title(product_title, X, item_mapper, item_inv_mapper, 
                                                          dict(zip(metadf['parent_asin'], metadf['title'])), 
                                                          k)

    # Get details of similar products from the metadata DataFrame
    similar_products = metadf[metadf['parent_asin'].isin(similar_product_ids)]

    # Extract 'large' images for each similar product
    similar_products['large_image_url'] = similar_products['images'].apply(extract_large_image)

    return render_template('main.html', similar_products=similar_products.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
