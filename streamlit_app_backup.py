import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handler import DataHandler
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model import SentimentModel
from topic_analyzer import TopicAnalyzer
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_preprocess_data():
    """Load and preprocess data with caching."""
    try:
        # Initialize components
        data_handler = DataHandler()
        preprocessor = TextPreprocessor()
        
        # Load data (try sample first, then full dataset)
        try:
            # First try to load from data folder, then from current directory
            data_path = "data/sentiment140.csv"
            if not os.path.exists(data_path):
                data_path = "sentiment140.csv"
            
            if not os.path.exists(data_path):
                st.error("Dataset not found! Please download the Sentiment140 dataset and place it in the project folder.")
                st.info("Download from: http://help.sentiment140.com/for-students")
                st.info("Place the file as 'sentiment140.csv' in the project root or 'data/' folder")
                return None, None
            
            df = data_handler.load_data(sample_size=50000)  # Load larger sample to avoid empty vocabulary
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.info("Please ensure the dataset file is properly formatted and accessible")
            return None, None
        
        # Preprocess data
        df_processed = preprocessor.preprocess_dataframe(df, 'text', 'cleaned_text')
        
        # Check if we have enough data after preprocessing
        if len(df_processed) < 1000:
            st.warning(f"Limited data after preprocessing: {len(df_processed)} rows. Using larger sample...")
            # Try loading more data if we have too little after preprocessing
            if len(df_processed) < 500:
                try:
                    df_large = data_handler.load_data(sample_size=100000)
                    df_processed = preprocessor.preprocess_dataframe(df_large, 'text', 'cleaned_text')
                    st.info(f"Reloaded with larger sample: {len(df_processed)} rows after preprocessing")
                except:
                    st.warning("Could not load larger sample. Continuing with current data.")
        
        # Additional check for empty text after preprocessing
        if df_processed['cleaned_text'].str.len().mean() < 5:
            st.warning("Very short average text length after preprocessing. Check preprocessing parameters.")
        
        return df_processed, preprocessor
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def train_models_and_analyze(df_processed, preprocessor, custom_topics):
    """Train models and perform analysis (not cached - allows dynamic topics)."""
    try:
        # Initialize components
        feature_engineer = FeatureEngineer()
        model = SentimentModel()
        topic_analyzer = TopicAnalyzer()
        visualizer = Visualizer()
        
        # Update topic analyzer with custom topics
        topic_analyzer.topic_keywords = custom_topics
        print(f"Using custom topics: {list(custom_topics.keys())}")
        
        # Feature engineering
        X = feature_engineer.fit_transform(df_processed['cleaned_text'])
        y = df_processed['sentiment']
        
        # Train model
        X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_split(X, y)
        model.train(X_train, y_train)
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test)
        
        # Topic sentiment analysis
        sentiment_results = topic_analyzer.get_topic_sentiment_analysis(
            df_processed, model, feature_engineer.vectorizer, 'text'
        )
        
        # Topic trends
        topic_trends = topic_analyzer.analyze_topic_trends(df_processed, 'cleaned_text')
        
        # Generate insights
        insights = topic_analyzer.generate_insights(sentiment_results)
        
        app_data = {
            'df': df_processed,
            'data_handler': DataHandler(),
            'preprocessor': preprocessor,
            'feature_engineer': feature_engineer,
            'model': model,
            'topic_analyzer': topic_analyzer,
            'visualizer': visualizer,
            'evaluation': evaluation,
            'sentiment_results': sentiment_results,
            'topic_trends': topic_trends,
            'insights': insights
        }
        
        return app_data
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

def load_data_and_models():
    """Load data and train models with smart caching."""
    try:
        # Get current topics
        custom_topics = st.session_state.get('custom_topics', {
            'Movies': ['movie', 'film', 'actor', 'actress', 'director', 'cinema', 'theater', 'hollywood', 'watch', 'screen'],
            'Sports': ['cricket', 'match', 'team', 'game', 'player', 'sport', 'football', 'basketball', 'tennis', 'ipl', 'world cup'],
            'Technology': ['phone', 'mobile', 'ai', 'app', 'software', 'computer', 'laptop', 'tech', 'digital', 'internet', 'device']
        })
        
        # Check if we have cached data and preprocessing
        cached_data_key = 'cached_df_processed'
        cached_preprocessor_key = 'cached_preprocessor'
        
        if cached_data_key not in st.session_state:
            # Load and preprocess data (cached)
            with st.spinner("Loading and preprocessing data..."):
                df_processed, preprocessor = load_and_preprocess_data()
                if df_processed is None:
                    return None
                
                # Cache the expensive preprocessing
                st.session_state[cached_data_key] = df_processed
                st.session_state[cached_preprocessor_key] = preprocessor
        else:
            # Use cached data
            df_processed = st.session_state[cached_data_key]
            preprocessor = st.session_state[cached_preprocessor_key]
            print("Using cached preprocessed data")
        
        # Train models and analyze (not cached to allow dynamic topics)
        with st.spinner("Training models and analyzing topics..."):
            app_data = train_models_and_analyze(df_processed, preprocessor, custom_topics)
        
        return app_data
        
    except Exception as e:
        st.error(f"Error loading data and models: {str(e)}")
        st.error("Please check your dataset and try again.")
        return None

def main():
    """Main application function."""
    
    # Initialize session state
    if 'app_data' not in st.session_state:
        st.session_state['app_data'] = None
    
    # Professional Sidebar Navigation
    st.sidebar.markdown("""
    <style>
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 0.5rem;
            text-align: center;
        }
        .nav-section {
            margin-bottom: 2rem;
        }
        .nav-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #6b7280;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Navigation sections
    st.sidebar.markdown('<div class="nav-section"><div class="nav-header">Main Pages</div></div>', unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "",
        ["Home", "Topic Analysis", "Comparison", "Insights"],
        key="main_navigation"
    )
    
    # Add dataset as separate section
    st.sidebar.markdown('<div class="nav-section"><div class="nav-header">Data Management</div></div>', unsafe_allow_html=True)
    if st.sidebar.button("Dataset Overview", use_container_width=True, key="dataset_btn"):
        st.session_state.main_navigation = "Dataset Overview"
        st.rerun()
    
    # Custom topic keywords input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Custom Topics")
    
    # Initialize custom topics in session state
    if 'custom_topics' not in st.session_state:
        st.session_state['custom_topics'] = {
            'Movies': ['movie', 'film', 'actor', 'actress', 'director', 'cinema', 'theater', 'hollywood', 'watch', 'screen'],
            'Sports': ['cricket', 'match', 'team', 'game', 'player', 'sport', 'football', 'basketball', 'tennis', 'ipl', 'world cup'],
            'Technology': ['phone', 'mobile', 'ai', 'app', 'software', 'computer', 'laptop', 'tech', 'digital', 'internet', 'device']
        }
    
    # Allow users to add custom topics
    with st.sidebar.expander("Add Custom Topic", expanded=False):
        topic_name = st.text_input("Topic Name:", key="new_topic_name")
        topic_keywords = st.text_area("Keywords (comma-separated):", key="new_topic_keywords", 
                                   help="Enter keywords separated by commas")
        
        if st.button("Add Topic", key="add_topic_btn"):
            if topic_name and topic_keywords:
                # Parse keywords
                keywords = [kw.strip().lower() for kw in topic_keywords.split(',')]
                keywords = [kw for kw in keywords if kw]  # Remove empty strings
                
                if keywords:
                    st.session_state['custom_topics'][topic_name] = keywords
                    st.success(f"Added topic '{topic_name}' with {len(keywords)} keywords")
                    st.rerun()
                else:
                    st.error("Please enter valid keywords")
            else:
                st.error("Please enter both topic name and keywords")
        
        # Show current topics
        st.markdown("**Current Topics:**")
        for topic, keywords in st.session_state['custom_topics'].items():
            st.markdown(f"- **{topic}**: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
    
    # Clear custom topics button
    if st.sidebar.button("Reset to Default Topics"):
        st.session_state['custom_topics'] = {
            'Movies': ['movie', 'film', 'actor', 'actress', 'director', 'cinema', 'theater', 'hollywood', 'watch', 'screen'],
            'Sports': ['cricket', 'match', 'team', 'game', 'player', 'sport', 'football', 'basketball', 'tennis', 'ipl', 'world cup'],
            'Technology': ['phone', 'mobile', 'ai', 'app', 'software', 'computer', 'laptop', 'tech', 'digital', 'internet', 'device']
        }
        st.success("Topics reset to default")
        st.rerun()
    
    # Check if we need to reload models (custom topics changed)
    topics_changed = False
    if 'last_topics_hash' not in st.session_state:
        st.session_state['last_topics_hash'] = str(hash(str(st.session_state.get('custom_topics', {}))))
    else:
        current_topics_hash = str(hash(str(st.session_state.get('custom_topics', {}))))
        if st.session_state['last_topics_hash'] != current_topics_hash:
            st.session_state['last_topics_hash'] = current_topics_hash
            topics_changed = True
            # Only clear app_data, keep cached preprocessing
            st.session_state['app_data'] = None
    
    # Load data and models if not already loaded or topics changed
    if st.session_state['app_data'] is None:
        app_data = load_data_and_models()
        
        if app_data is None:
            st.error("Failed to load application data. Please check the dataset and try again.")
            st.stop()
        else:
            st.session_state['app_data'] = app_data
            if topics_changed:
                st.success("Models updated with new topics!")
    else:
        app_data = st.session_state['app_data']
    
    # Extract components
    df = app_data['df']
    sentiment_results = app_data['sentiment_results']
    topic_trends = app_data['topic_trends']
    insights = app_data['insights']
    model = app_data['model']
    feature_engineer = app_data['feature_engineer']
    topic_analyzer = app_data['topic_analyzer']
    visualizer = app_data['visualizer']
    
    # Route to appropriate page
    try:
        if page == "Home":
            from pages.home import show_home_page
            show_home_page()
        elif page == "Dataset Overview":
            from pages.dataset import show_dataset_page
            show_dataset_page()
        elif page == "Topic Analysis":
            from pages.topic_analysis import show_topic_analysis_page
            show_topic_analysis_page()
        elif page == "Comparison":
            from pages.comparison import show_comparison_page
            show_comparison_page()
        elif page == "Insights":
            from pages.insights import show_insights_page
            show_insights_page()
    except ImportError as e:
        st.error(f"Error importing page modules: {str(e)}")
        st.error("Please ensure all page files are properly created in the pages/ directory.")
    except Exception as e:
        st.error(f"Error displaying page: {str(e)}")
        st.info("Please try refreshing the page or check the application logs.")

if __name__ == "__main__":
    main()
