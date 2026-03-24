import streamlit as st  
import pandas as pd  
import numpy as np 
import pickle  
from pathlib import Path  
import sys  
sys.path.append(str(Path(__file__).parent / 'models'))

st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    /* Success message styling */
    .success-msg {
        color: #4CAF50;
        font-weight: bold;
    }
    
    /* Warning message styling */
    .warning-msg {
        color: #FF9800;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data  # Cache data to avoid reloading on every interaction
def load_data_and_models():
    """
    Load all saved models and processed data.
    
    This function loads:
    - Processed book dataframe
    - TF-IDF vectorizer
    - TF-IDF matrix
    - Cosine similarity matrix
    - K-means clustering model
    
    Returns:
        tuple: (df, tfidf_vectorizer, tfidf_matrix, cosine_sim, kmeans)
    """
    try:
        # Define paths to model files
        models_path = Path('models')
        df = pd.read_pickle(models_path / 'books_data.pkl')
        with open(models_path / 'tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(models_path / 'tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(models_path / 'cosine_similarity_matrix.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        
        # Load K-means clustering model
        # Used for cluster-based recommendations
        with open(models_path / 'kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        return df, tfidf_vectorizer, tfidf_matrix, cosine_sim, kmeans
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()  # Stop execution if models can't be loaded

# Load all data and models at startup
df_books, tfidf_vectorizer, tfidf_matrix, cosine_similarity_matrix, kmeans_model = load_data_and_models()



def get_content_based_recommendations(book_name, n_recommendations=10):
    """
    Generate recommendations using content-based filtering.
    
    This method finds books similar to the input book based on:
    - Text content (titles, descriptions)
    - Using cosine similarity between TF-IDF vectors
    
    Args:
        book_name (str): Name of the book to find similar books for
        n_recommendations (int): Number of recommendations to return
    
    Returns:
        pd.DataFrame: Recommended books with similarity scores
    """
    try:
        book_indices = df_books[df_books['Book Name'].str.lower() == book_name.lower()].index
        
        if len(book_indices) == 0:
            book_indices = df_books[df_books['Book Name'].str.lower().str.contains(book_name.lower(), na=False)].index
            if len(book_indices) == 0:
                return None, f"Book '{book_name}' not found in database."
        
        book_idx = book_indices[0]
        print(book_idx)

        sim_scores = list(enumerate(cosine_similarity_matrix[book_idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        book_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = df_books.iloc[book_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        recommendations['similarity_percentage'] = (recommendations['similarity_score'] * 100).round(2)
        
        return recommendations, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_cluster_based_recommendations(book_name, n_recommendations=10):
    """
    Generate recommendations using cluster-based approach.
    
    This method recommends books from the same cluster as the input book.
    Books in the same cluster share similar themes, topics, or characteristics.
    
    Args:
        book_name (str): Name of the book
        n_recommendations (int): Number of recommendations to return
    
    Returns:
        pd.DataFrame: Recommended books from the same cluster
    """
    try:
        # Find the book
        book_indices = df_books[df_books['Book Name'].str.lower() == book_name.lower()].index
        
        if len(book_indices) == 0:
            book_indices = df_books[df_books['Book Name'].str.lower().str.contains(book_name.lower(), na=False)].index
            if len(book_indices) == 0:
                return None, f"Book '{book_name}' not found in database."
        
        book_idx = book_indices[0]
        
        # Get the cluster ID of the input book
        book_cluster = df_books.iloc[book_idx]['Cluster']
        
        # Find all books in the same cluster (excluding the input book)
        cluster_books = df_books[
            (df_books['Cluster'] == book_cluster) & 
            (df_books.index != book_idx)
        ].copy()
        

        if 'Rating' in cluster_books.columns:
            cluster_books = cluster_books.nlargest(n_recommendations, 'Rating')
        else:
            cluster_books = cluster_books.head(n_recommendations)
        
        return cluster_books, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_hybrid_recommendations(book_name, n_recommendations=10, 
                               weight_content=0.5, weight_cluster=0.3):
    """
    Generate recommendations using hybrid approach.
    
    Combines multiple factors:
    - Content similarity (50% weight by default)
    - Cluster membership (30% weight)
    - Popularity score (20% weight)
    
    Args:
        book_name (str): Name of the book
        n_recommendations (int): Number of recommendations
        weight_content (float): Weight for content similarity
        weight_cluster (float): Weight for cluster membership
        weight_popularity (float): Weight for popularity
    
    Returns:
        pd.DataFrame: Recommended books with hybrid scores
    """
    try:
        # Find the book
        book_indices = df_books[df_books['Book Name'].str.lower() == book_name.lower()].index
        
        if len(book_indices) == 0:
            book_indices = df_books[df_books['Book Name'].str.lower().str.contains(book_name.lower(), na=False)].index
            if len(book_indices) == 0:
                return None, f"Book '{book_name}' not found in database."
        
        book_idx = book_indices[0]
        book_cluster = df_books.iloc[book_idx]['Cluster']
        
        # Create scoring dataframe
        df_score = df_books.copy()
        
        # 1. Content similarity score (from cosine similarity matrix)
        df_score['content_score'] = cosine_similarity_matrix[book_idx]
        
        # 2. Cluster score (1 if same cluster, 0 otherwise)
        df_score['cluster_score'] = (df_score['Cluster'] == book_cluster).astype(int)
        
        
        # Calculate weighted hybrid score
        df_score['hybrid_score'] = (
            weight_content * df_score['content_score'] +
            weight_cluster * df_score['cluster_score'] 
        )
        
        # Remove the input book
        df_score = df_score[df_score.index != book_idx]
        
        # Get top recommendations
        recommendations = df_score.nlargest(n_recommendations, 'hybrid_score')
        
        return recommendations, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_popular_books(n_recommendations=10, rating_threshold=4.0):
    """
    Get the most popular books overall.
    
    Useful for:
    - New users (cold start problem)
    - Discovering trending books
    - Getting started with the system
    
    Args:
        n_recommendations (int): Number of books to return
        rating_threshold (float): Minimum rating filter
    
    Returns:
        pd.DataFrame: Popular books
    """
    try:
        # Filter by rating threshold
        if 'Rating' in df_books.columns:
            popular_books = df_books[df_books['Rating'] >= rating_threshold].copy()
        else:
            popular_books = df_books.copy()
        
        if 'Rating' in popular_books.columns:
            popular_books = popular_books.nlargest(n_recommendations, 'Rating')
        else:
            popular_books = popular_books.head(n_recommendations)
        
        return popular_books, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# MAIN APP INTERFACE
# ============================================================================

def app():
    st.markdown('<h1 class="main-title">📚 Audible Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Book Recommendation System</p>', unsafe_allow_html=True)


    
    # Display dataset statistics in sidebar
    st.sidebar.subheader("📈 Dataset Statistics")
    st.sidebar.metric("Total Books", f"{len(df_books):,}")
    st.sidebar.metric("Unique Authors", f"{df_books['Author'].nunique():,}")
    st.sidebar.metric("Number of Clusters", f"{df_books['Cluster'].nunique()}")
    
    if 'Rating' in df_books.columns:
        avg_rating = df_books['Rating'].mean()
        st.sidebar.metric("Average Rating", f"{avg_rating:.2f} ⭐")

    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "🌟 Popular Books", "🎲 Random Discovery"])
        
    with tab1:
        st.header("Find Books Similar to Your Favorites")
        st.markdown("""
        Enter a book name to get personalized recommendations using our advanced algorithms.
        We offer three recommendation methods:
        - **Content-Based**: Based on book descriptions and content
        - **Cluster-Based**: Based on similar themes and topics
        - **Hybrid**: Combines multiple factors for best results
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:

            book_name_input = st.selectbox(
                "Select or type a book name:",
                options=[''] + sorted(df_books['Book Name'].unique().tolist()),
                help="Start typing to search for books"
            )
        
        with col2:
            n_recs = st.slider(
                "Number of recommendations:",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="How many recommendations do you want?"
            )
        
        rec_method = st.radio(
            "Choose recommendation method:",
            ["Hybrid (Recommended)", "Content-Based", "Cluster-Based"],
            horizontal=True,
            help="Hybrid combines multiple methods for best results"
        )
        
        with st.expander("⚙️ Advanced Options (Hybrid Method Only)"):
            st.markdown("Adjust weights for the hybrid recommendation algorithm:")
            
            col_w1, col_w2 = st.columns(2)
            
            with col_w1:
                weight_content = st.slider(
                    "Content Weight",
                    0.0, 1.0, 0.5, 0.1,
                    help="Importance of content similarity"
                )
            
            with col_w2:
                weight_cluster = st.slider(
                    "Cluster Weight",
                    0.0, 1.0, 0.3, 0.1,
                    help="Importance of cluster membership"
                )
            

            
            total_weight = weight_content + weight_cluster
            if total_weight > 0:
                weight_content /= total_weight
                weight_cluster /= total_weight
        
        # Generate recommendations button
        if st.button("🚀 Get Recommendations", type="primary"):
            if book_name_input and book_name_input != '':
                with st.spinner("Analyzing and finding similar books..."):
                    # Get recommendations based on selected method
                    if rec_method == "Content-Based":
                        recommendations, error = get_content_based_recommendations(
                            book_name_input, n_recs
                        )
                    elif rec_method == "Cluster-Based":
                        recommendations, error = get_cluster_based_recommendations(
                            book_name_input, n_recs
                        )
                    else:  # Hybrid
                        recommendations, error = get_hybrid_recommendations(
                            book_name_input, n_recs,
                            weight_content, weight_cluster
                        )
                    
                    # Display results
                    if error:
                        st.error(error)
                    elif recommendations is not None and len(recommendations) > 0:
                        st.success(f"Found {len(recommendations)} recommendations for '{book_name_input}'")
                        
                        # Display recommendations in a nice format
                        st.markdown("### 📚 Recommended Books:")
                        
                        for idx, row in recommendations.iterrows():
                            # Create an expander for each book
                            with st.expander(f"**{row['Book Name']}** by {row['Author']}", expanded=(idx == recommendations.index[0])):
                                # Create columns for book details
                                detail_col1, detail_col2, detail_col3 = st.columns([2, 1, 1])
                                
                                with detail_col1:
                                    if 'Description' in row and pd.notna(row['Description']) and row['Description'] != '':
                                        st.markdown(f"**Description:** {row['Description'][:200]}...")
                                    else:
                                        st.markdown("*No description available*")
                                
                                with detail_col2:
                                    if 'Rating' in row:
                                        st.metric("Rating", f"{row['Rating']:.2f} ⭐")
                                    if 'Price' in row:
                                        st.metric("Price", f"{row['Price']:.2f}")
                                
                                with detail_col3:
                                    if 'similarity_score' in row:
                                        st.metric("Similarity", f"{row['similarity_score']:.2%}")
                                    if 'hybrid_score' in row:
                                        st.metric("Match Score", f"{row['hybrid_score']:.2f}")
                                    if 'cluster' in row:
                                        st.metric("Cluster", int(row['cluster']))
                    else:
                        st.warning("No recommendations found. Try a different book.")
            else:
                st.warning("⚠️ Please select a book name first.")
        
        
        with tab2:
            st.header("🌟 Trending & Popular Books")
            st.markdown("Discover the most popular and highly-rated books in our collection.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_popular = st.slider(
                    "Number of books to show:",
                    5, 30, 15, 5
                )
            
            with col2:
                rating_filter = st.slider(
                    "Minimum rating:",
                    0.0, 5.0, 4.0, 0.5
                )
            
            if st.button("📊 Show Popular Books"):
                popular_books, error = get_popular_books(n_popular, rating_filter)
                
                if error:
                    st.error(error)
                elif popular_books is not None and len(popular_books) > 0:
                    st.success(f"Showing top {len(popular_books)} popular books")
                    
                    # Display as a formatted table
                    display_cols = ['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']
                    display_cols = [col for col in display_cols if col in popular_books.columns]
                    
                    # Format the dataframe for better display
                    display_df = popular_books[display_cols].copy()
                    display_df = display_df.reset_index(drop=True)
                    display_df.index = display_df.index + 1  # Start index from 1
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.warning("No books found with the specified criteria.")
        
        # ====================================================================
        # TAB 3: RANDOM DISCOVERY
        # ====================================================================
        
    with tab3:
        st.header("🎲 Random Book Discovery")
        st.markdown("Feeling adventurous? Click the button to discover random books!")
        
        n_random = st.slider("Number of random books:", 3, 10, 5, 1)
        
        if st.button("🎲 Discover Random Books"):
            # Get random books
            random_books = df_books.sample(n=n_random)
            
            st.success(f"Here are {n_random} random book picks for you!")
            
            for idx, row in random_books.iterrows():
                with st.container():
                    st.markdown(f"### 📖 {row['Book Name']}")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Author:** {row['Author']}")
                        if 'Description' in row and pd.notna(row['Description']):
                            st.markdown(f"*{row['Description'][:150]}...*")
                    
                    with col2:
                        if 'Rating' in row:
                            st.metric("Rating", f"{row['Rating']:.2f} ⭐")
                        if 'Price' in row:
                            st.metric("Price", f"{row['Price']:.2f}")
                    
                    st.markdown("---")
    
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "📚 Audible Insights | Book Recommendation System | 2024"
        "</div>",
        unsafe_allow_html=True
    )