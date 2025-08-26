import streamlit as st
import json

st.title("👗 Product Catalog")
st.write("Browse our collection of clothing products")

import src.config as CONFIG

# Load data
@st.cache_data
def load_products():
    try:
        with open(CONFIG.DATASET_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return []

def display_product_card(product):
    """Display a single product in card format"""
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{product['name']} - ${product['price']}")
            st.caption(f"{product['brand']} • {product['category'].title()} • {product['color']}")
            st.write(product['description'])
        
        with col2:
            st.write(f"**Material:** {product['material']}")
            st.write(f"**Sizes:** {', '.join(product['size'])}")
            
            # View product button
            if st.button("View Details", key=f"btn_{product['id']}"):
                st.query_params.product_id = product['id']
                st.rerun()
        
        st.divider()

def main():
    # Check for single product view
    product_id = st.query_params.get('product_id')
    products = load_products()
    
    if product_id:
        # Single product view
        product = next((p for p in products if p['id'] == int(product_id)), None)
        if product:
            if st.button("← Back to Catalog"):
                del st.query_params.product_id
                st.rerun()
            
            st.title(product['name'])
            st.write(f"**{product['brand']}** • ${product['price']} • {product['color']}")
            st.write(product['description'])
            st.write(f"**Material:** {product['material']}")
            st.write(f"**Available Sizes:** {', '.join(product['size'])}")
        else:
            st.error("Product not found")
    else:
        # Catalog view
        st.title("👗 Product Catalog")
        
        # Simple search
        search = st.text_input("🔍 Search:", placeholder="Product name or brand...")
        
        # Filter products
        if search:
            products = [p for p in products if 
                       search.lower() in p['name'].lower() or 
                       search.lower() in p['brand'].lower()]
        
        st.write(f"**{len(products)} products**")
        
        # Display products
        for product in products:
            display_product_card(product)

if __name__ == "__main__":
    main()