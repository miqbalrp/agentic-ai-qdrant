import streamlit as st
import json

st.title("👗 Product Catalog")
st.write("Browse our collection of clothing products")

# Load data
@st.cache_data
def load_products():
    try:
        with open("dataset/ecommerce_product.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return []

products = load_products()

if not products:
    st.error("No products found.")
    st.stop()

# Simple search
search = st.text_input("Search products:", placeholder="Search by name or brand...")

# Filter products if search term provided
if search:
    products = [p for p in products if 
                search.lower() in p['name'].lower() or 
                search.lower() in p['brand'].lower()]

st.write(f"**{len(products)} products found**")

# Display products
for product in products:
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(product['name'])
            st.write(f"*{product['brand']}* • {product['category'].title()}")
            st.write(product['description'])
        
        with col2:
            st.metric("Price", f"${product['price']:.2f}")
            st.write(f"**Color:** {product['color']}")
            st.write(f"**Material:** {product['material']}")
        
        with col3:
            st.write("**Available Sizes:**")
            st.write(", ".join(product['size']))
        
        st.markdown("---")