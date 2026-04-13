import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handler import DataHandler
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model import SentimentModel
from topic_analyzer import TopicAnalyzer
from visualization import Visualizer

def show_home_page():
    """Display the home page content."""
    
    st.markdown('<h1 class="main-header">📊 Social Media Trend and Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This application analyzes Twitter sentiment across different topics using Machine Learning.
    
    ### Features:
    - **Sentiment Analysis**: Classify tweets as Positive or Negative
    - **Topic Identification**: Automatically categorize tweets into Movies, Sports, or Technology
    - **Trend Analysis**: Identify trending keywords and topics
    - **Interactive Visualizations**: Charts and word clouds for better understanding
    
    ### How to Use:
    1. Navigate through different pages using the sidebar
    2. Explore dataset statistics and sample data
    3. Analyze sentiment for specific topics
    4. Compare topics side by side
    5. View automatically generated insights
    
    ### 📈 Model Performance:
    """)
    
    # Load cached data
    app_data = st.session_state.get('app_data')
    if app_data:
        # Display model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{app_data['evaluation']['accuracy']:.4f}")
        with col2:
            st.metric("Cross-Val Score", f"{app_data['evaluation']['cv_mean']:.4f}")
        with col3:
            st.metric("Total Tweets Analyzed", len(app_data['df']))
        
        # Quick insights
        st.markdown("### 🔍 Quick Insights")
        for insight in app_data['insights'][:3]:  # Show top 3 insights
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Sentiment prediction demo
        st.markdown("### 🤖 Try Sentiment Prediction")
        user_input = st.text_area("Enter a tweet to analyze sentiment:", 
                                 placeholder="Type your tweet here...")
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                try:
                    # Preprocess and predict
                    cleaned_text = app_data['preprocessor'].preprocess_text(user_input)
                    sentiment, confidence = app_data['model'].predict_sentiment(
                        cleaned_text, app_data['feature_engineer'].vectorizer
                    )
                    topic = app_data['topic_analyzer'].identify_topic(user_input)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col3:
                        st.metric("Topic", topic)
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    else:
        st.warning("Please wait for data to load or refresh the page.")

if __name__ == "__main__":
    show_home_page()
