import streamlit as st
import asyncio
import time

import src.shopping_agent as shopping_agent

st.set_page_config(
    page_title="Shopping Chat Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Header
st.title("🛍️ AI Shopping Chat Assistant")
st.markdown("Have a conversation with your personal shopping assistant")

# Example queries for inspiration
with st.expander("💡 Example conversations"):
    st.markdown("""
    - "I need a blue dress for a wedding"
    - "Show me comfortable jeans from Levi's"  
    - "Looking for a warm winter sweater under $50"
    - "What about something in red instead?"
    - "Can you show me similar items but cheaper?"
    """)

# Display chat history
st.markdown("### Conversation")
chat_container = st.container()

with chat_container:
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    else:
        st.markdown("*Start a conversation by typing your question below...*")

# Chat input
user_input = st.chat_input(
    "Ask me about clothing items...",
    disabled=st.session_state.is_processing
)

# Process new message
if user_input and not st.session_state.is_processing:
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Set processing state
    st.session_state.is_processing = True
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Show typing indicator and process
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create conversation context by joining recent messages
                conversation_context = ""
                if len(st.session_state.chat_history) > 1:
                    # Include last few messages for context
                    recent_messages = st.session_state.chat_history[-3:]  # Last 3 messages
                    for msg in recent_messages[:-1]:  # Exclude the current message
                        if msg['role'] == 'user':
                            conversation_context += f"User: {msg['content']}\n"
                        else:
                            conversation_context += f"Assistant: {msg['content']}\n"
                    conversation_context += f"User: {user_input}"
                else:
                    conversation_context = user_input
                
                # Run the async function with context
                result = asyncio.run(shopping_agent.run_agent(conversation_context))
                
                # Display assistant response
                st.write(result)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant', 
                    'content': result
                })
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"❌ {error_msg}"
                })
    
    # Reset processing state
    st.session_state.is_processing = False
    
    # Rerun to update the interface
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by [OpenAI API](https://openai.com/) • [OpenAI Agents SDK](https://github.com/openai/openai-agents) • [Qdrant](https://qdrant.tech/) • [Streamlit](https://streamlit.io/)*")