import streamlit as st
import asyncio
import time

st.set_page_config(
    page_title="Shopping Assistant",
    page_icon="🛍️",
    layout="centered"
)

# Header
st.title("🛍️ AI Shopping Assistant")
st.markdown("Find the perfect clothing items with AI-powered search")

# Example queries for inspiration
with st.expander("💡 Example searches"):
    st.markdown("""
    - "I need a blue dress for a wedding"
    - "Show me comfortable jeans from Levi's"
    - "Looking for a warm winter sweater"
    - "Casual shirts under $40"
    - "Athletic wear for running"
    """)

# Main search interface
user_query = st.text_input(
    "What are you looking for?",
    placeholder="Describe what you want to find...",
    help="Try being specific about color, brand, style, or occasion"
)

# Search button with better styling
if st.button("🔍 Search", type="primary", use_container_width=True):
    if user_query.strip():
        with st.spinner("Searching our products..."):
            try:
                # Import here to avoid issues if module doesn't exist
                import src.shopping_agent as shopping_agent
                
                # Run the async function
                result = asyncio.run(shopping_agent.run_agent(user_query))
                
                # Display results
                st.success("Found results!")
                st.markdown("### Results:")
                st.write(result)
                
            except ImportError:
                st.error("Shopping agent module not found. Please check your setup.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter a search query.")

# Recent searches (simple session state)
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

# Add current search to recent searches
if user_query and user_query.strip() and st.button:
    if user_query not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, user_query)
        # Keep only last 5 searches
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

# Footer
st.markdown("---")
st.markdown("*Powered by [OpenAI API](https://openai.com/) • [OpenAI Agents SDK](https://github.com/openai/openai-agents) • [Qdrant](https://qdrant.tech/) • [Streamlit](https://streamlit.io/)")