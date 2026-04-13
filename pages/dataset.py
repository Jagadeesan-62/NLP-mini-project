import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_dataset_page():
    """Display the dataset overview page."""
    
    st.markdown('<h1 class="main-header">📋 Dataset Overview</h1>', unsafe_allow_html=True)
    
    # Load cached data
    app_data = st.session_state.get('app_data')
    if not app_data:
        st.warning("Please wait for data to load or refresh the page.")
        return
    
    df = app_data['df']
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Statistics")
        info = app_data['data_handler'].get_data_info()
        
        st.metric("Total Tweets", info['shape'][0])
        st.metric("Columns", info['shape'][1])
        st.metric("Positive Tweets", info['sentiment_distribution'].get(1, 0))
        st.metric("Negative Tweets", info['sentiment_distribution'].get(0, 0))
    
    with col2:
        st.markdown("### 📈 Sentiment Distribution")
        sentiment_dist = info['sentiment_distribution']
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Positive', 'Negative']
        sizes = [sentiment_dist.get(1, 0), sentiment_dist.get(0, 0)]
        colors = ['#2ecc71', '#e74c3c']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    # Sample data
    st.markdown("### 🔍 Sample Data")
    sample_df = df[['text', 'sentiment']].head(10)
    sample_df['sentiment'] = sample_df['sentiment'].map({0: 'Negative', 1: 'Positive'})
    st.dataframe(sample_df, use_container_width=True)
    
    # Preprocessing statistics
    st.markdown("### 🧹 Preprocessing Statistics")
    stats = app_data['preprocessor'].get_preprocessing_stats(
        app_data['data_handler'].data, df
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Count", stats['original_count'])
    with col2:
        st.metric("Processed Count", stats['processed_count'])
    with col3:
        st.metric("Removal Rate", f"{stats['removal_percentage']:.1f}%")
    
    # Data quality metrics
    st.markdown("### 📊 Data Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text length distribution
        df_copy = df.copy()
        df_copy['text_length'] = df_copy['text'].astype(str).str.len()
        df_copy['cleaned_length'] = df_copy['cleaned_text'].astype(str).str.len()
        
        st.metric("Avg Original Text Length", f"{df_copy['text_length'].mean():.1f} chars")
        st.metric("Avg Cleaned Text Length", f"{df_copy['cleaned_length'].mean():.1f} chars")
        st.metric("Text Length Reduction", f"{((df_copy['text_length'].mean() - df_copy['cleaned_length'].mean()) / df_copy['text_length'].mean() * 100):.1f}%")
    
    with col2:
        # Topic distribution
        topic_dist = app_data['topic_analyzer'].get_topic_distribution(df)
        
        st.markdown("**Topic Distribution:**")
        for topic, count in topic_dist.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                st.write(f"- {topic}: {count} tweets ({percentage:.1f}%)")
    
    # Missing values analysis
    st.markdown("### 🔍 Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_df = missing_data[missing_data > 0].reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")

if __name__ == "__main__":
    show_dataset_page()
