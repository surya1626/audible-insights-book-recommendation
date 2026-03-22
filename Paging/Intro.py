import streamlit as st

def app():

    st.markdown("""
        ## 📚 Audible Insights - Book Recommendation System
        
        ### Project Overview
        This intelligent book recommendation system helps users discover books they'll love using advanced
        machine learning and natural language processing techniques.
        
        ### 🎯 Features
        
        #### 1. **Multiple Recommendation Methods**
        - **Content-Based Filtering**: Recommends books based on similarity in content, descriptions, and themes
        - **Cluster-Based Recommendations**: Groups similar books together and recommends from the same cluster
        - **Hybrid Approach**: Combines multiple methods for the most accurate recommendations
        - **Popular Books**: Discover trending and highly-rated books
        
        #### 2. **Advanced NLP Processing**
        - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
        - Text preprocessing and cleaning
        - Cosine similarity calculation
        
        #### 3. **Machine Learning Models**
        - K-Means clustering for book grouping
        - PCA for dimensionality reduction
        - Similarity-based matching algorithms
        
        ### 🛠️ Technology Stack
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Pandas & NumPy**: Data manipulation
        - **Scikit-learn**: Machine learning algorithms
        - **NLTK**: Natural language processing
        
        
        ### 🚀 How It Works
        
        1. **Data Collection**: Books are loaded from Audible catalog datasets
        2. **Preprocessing**: Clean data, handle missing values, merge datasets
        3. **Feature Engineering**: Create new features like popularity scores
        4. **NLP Processing**: Convert text to numerical features using TF-IDF
        5. **Clustering**: Group similar books using K-Means
        6. **Recommendation**: Multiple algorithms provide personalized suggestions

        """)
    