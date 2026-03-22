import streamlit as st


st.set_page_config(
    page_title="Audible Insights - Book Recommender",
    page_icon="📚",  
    layout="wide", 
    initial_sidebar_state="expanded"  
)


st.sidebar.title("📚 Audible Insights")


if st.sidebar.button("🔄 Hard Refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

pages = {
    "intro": "🏠 Introduction",
    "insights": "📚 Book Recommender",
    "creator": "👨🏻‍💻About Creator"
}

page = st.sidebar.radio(
    "Go to",
    options=list(pages.keys()),
    format_func=lambda x: pages[x]
)

if page == "intro":
    import Paging.Intro as index
    index.app()

elif page == "insights":
    import Paging.insights as insights
    insights.app()

elif page == "creator":
    import Paging.creator as creator
    creator.app()
