import streamlit as st
from templates import demo_data, demo_model

# Set page config
st.set_page_config(
    page_title="Roof Rate",
    page_icon="🏠",
)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Create a sidebar for navigation
st.sidebar.title("Navigation")

# Use the session state to determine the default selection
page_options = ["Home", "Demo Model", "Demo Data - Coming Soon"]
default_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0

# Radio button for navigation
selected_page = st.sidebar.radio("Go to", page_options, index=default_index)

# Update session state when radio changes
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

# Display the selected page based on session state
if st.session_state.current_page == "Home":
    st.title("Welcome to Roof Rate")
    st.markdown(
        """
        ### Choose a page from the sidebar:
        - **Demo Model**: Try our roof analysis model
        - **Demo Data**: Coming Soon
        """
    )
    
    # Home button in Demo Model redirects to home
    if st.button("Go to Demo Model", key="home_to_demo"):
        st.session_state.current_page = "Demo Model"
        st.rerun()
    
elif st.session_state.current_page == "Demo Model":
    demo_model.show()
    
elif st.session_state.current_page == "Demo Data - Coming Soon":
    # demo_data.show()
    st.title("Demo Data - Coming Soon")
    st.write("This feature is currently under development.")

