from flask import Flask, render_template, request
import pandas as pd
import ast
from helper import most_popular_products, create_X_custom, find_similar_products_by_title, content_based_recommendations
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load your dataset
metadf = pd.read_csv('artifacts/meta_final.csv')
reviewdf = pd.read_csv('artifacts/filtered_dataset.csv')

# Function to extract 'large' image from the 'Images' column
def extract_large_image(image_column):
    try:
        image_dict = ast.literal_eval(image_column)
        return image_dict.get('large', None)  
    except (ValueError, SyntaxError):
        return None

# Prepare the user-item matrix for collaborative filtering
X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper = create_X_custom(reviewdf)

@app.route('/')
def home():
    popular_products = most_popular_products(metadf)
    popular_products['large_image_url'] = popular_products['images'].apply(extract_large_image)
    popular_products_list = popular_products.to_dict(orient='records')
    return render_template('index.html', popular_products=popular_products_list)

@app.route('/main')
def main():
    # Randomly select 100 product titles
    product_titles = metadf['title'].sample(n=100).tolist()
    return render_template('main.html', product_titles=product_titles)

@app.route('/recommendations', methods=['POST'])
def recommend():
    product_titles = metadf['title'].sample(n=100).tolist()
    product_title = request.form['prod']  # Get the product title from the form
    k = int(request.form.get('nbr', 5))  # Default to 5 recommendations if not specified
    filter_type = request.form['filter_type']  # Get the selected filtering type

    if filter_type == 'content':
        # Content-based filtering
        recommended_products = content_based_recommendations(metadf, product_title, top_n=k)
    else:
        # Collaborative filtering
        similar_product_ids = find_similar_products_by_title(
            product_title, X, item_mapper, item_inv_mapper, 
            dict(zip(metadf['parent_asin'], metadf['title'])), k
        )
        recommended_products = metadf[metadf['parent_asin'].isin(similar_product_ids)]

    # Extract 'large' images for each similar product
    recommended_products['large_image_url'] = recommended_products['images'].apply(extract_large_image)

    if recommended_products.empty:
        message = f"No products found similar to '{product_title}'. Please try another search."
        return render_template('main.html', similar_products=None, message=message, product_titles=product_titles)
    else:
        return render_template('main.html', similar_products=recommended_products.to_dict(orient='records'), message=None, product_titles=product_titles)

if __name__ == '__main__':
    app.run(debug=True)
