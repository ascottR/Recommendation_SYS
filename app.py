from flask import Flask, render_template, request,jsonify
import pandas as pd
import ast
from helper import most_popular_products, create_X_custom, find_similar_products_by_title, content_based_recommendations, most_popular_products_by_category,get_product_by_asin
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
     # Display most popular products overall by default
    popular_products = most_popular_products(metadf)
    popular_products['large_image_url'] = popular_products['images'].apply(extract_large_image)
    popular_products_list = popular_products.to_dict(orient='records')

        # Extract and flatten categories from the 'categories' column
    all_categories = []

    for cat in metadf['categories']:
        try:
            # Convert string representation of list back to an actual list
            category_list = ast.literal_eval(cat)
            all_categories.extend(category_list)
        except (ValueError, SyntaxError):
            continue  # Skip if conversion fails

    # Remove duplicates and sort the categories
    unique_categories = sorted(set([cat.strip() for cat in all_categories]))

    return render_template('index.html', popular_products=popular_products_list, categories=unique_categories)

@app.route('/trending', methods=['GET'])
def trending_by_category():
    category = request.args.get('category', None)  # Get the category from the query parameter
    print(category)
    if category:
        trending_products = most_popular_products_by_category(metadf, category)
    else:
        # Fallback: if no category is selected, show overall popular products
        trending_products = most_popular_products(metadf)

    # Extract 'large' images for each product
    trending_products['large_image_url'] = trending_products['images'].apply(extract_large_image)
    products_list = trending_products.to_dict(orient='records')
    print("Products List:", products_list)
    return jsonify(products_list)
    
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

@app.route('/product/<parent_asin>')
def product_detail(parent_asin):
    # Fetch product details using parent_asin from your data source
    product = get_product_by_asin(parent_asin, metadf)

    # Extract 'large' images for each product
    if product:
        # Directly extract the large image URL for this single product
        product['large_image_url'] = extract_large_image(product['images'])

    product_title = product['title']  # Get the product title 
    k = 3  # Default to 5 recommendations if not specified

    similar_product_ids = find_similar_products_by_title(product_title, X, item_mapper, item_inv_mapper, 
                                                          dict(zip(metadf['parent_asin'], metadf['title'])), 
                                                          k)

    # Get details of similar products from the metadata DataFrame
    similar_products = metadf[metadf['parent_asin'].isin(similar_product_ids)]

    # Extract 'large' images for each similar product
    similar_products['large_image_url'] = similar_products['images'].apply(extract_large_image)

    return render_template('productDetail.html', product=product , similar_products=similar_products.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
