import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def filter_tweets_by_keyword(df, keyword, text_column='text'):
    """Filter tweets containing the keyword."""
    if not keyword or not isinstance(keyword, str):
        return pd.DataFrame()
    
    # Clean keyword and create pattern
    keyword = keyword.strip().lower()
    pattern = r'\b' + re.escape(keyword) + r'\b'
    
    # Filter tweets containing the keyword
    mask = df[text_column].str.lower().str.contains(pattern, na=False, regex=True)
    filtered_df = df[mask].copy()
    
    return filtered_df

def analyze_keyword_sentiment(filtered_df, model, vectorizer, text_column='text'):
    """Analyze sentiment for filtered tweets."""
    if len(filtered_df) == 0:
        return {
            'total_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'positive_percentage': 0.0,
            'negative_percentage': 0.0
        }
    
    # Predict sentiment
    cleaned_texts = filtered_df['cleaned_text'].fillna('')
    text_vectors = vectorizer.transform(cleaned_texts)
    predictions = model.predict(text_vectors)
    
    # Count sentiments
    positive_count = sum(predictions == 1)
    negative_count = sum(predictions == 0)
    total_count = len(filtered_df)
    
    return {
        'total_count': total_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': (positive_count / total_count) * 100,
        'negative_percentage': (negative_count / total_count) * 100
    }

def get_top_keywords_for_tweets(filtered_df, text_column='cleaned_text', top_n=10):
    """Extract top keywords from filtered tweets."""
    if len(filtered_df) == 0:
        return []
    
    # Combine all text
    all_text = ' '.join(filtered_df[text_column].fillna('').astype(str))
    
    # Tokenize and count
    words = all_text.lower().split()
    word_freq = Counter(words)
    
    # Remove common stopwords
    stopwords = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'i', 'it', 'for', 'of', 'in', 'this', 'that', 'was', 'with', 'as', 'be', 'are', 'have', 'had', 'but', 'not', 'you', 'was', 'were', 'he', 'she', 'they', 'we', 'so', 'if', 'or', 'by', 'an', 'will', 'my', 'your', 'can', 'has', 'been', 'would', 'there', 'their', 'what', 'when', 'who', 'how', 'all', 'about', 'up', 'out', 'do', 'them', 'then', 'than', 'some', 'time', 'very', 'just', 'get', 'go', 'me', 'no', 'like', 'know', 'make', 'take', 'see', 'come', 'think', 'back', 'well', 'only', 'even', 'way', 'good', 'new', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'after', 'work', 'life', 'here', 'own', 'without', 'again', 'off', 'old', 'while', 'still', 'over', 'between', 'many', 'before', 'through', 'being', 'much', 'where', 'too', 'more'}
    
    # Filter out stopwords and short words
    filtered_words = [(word, count) for word, count in word_freq.items() 
                      if len(word) > 2 and word not in stopwords and not word.isdigit()]
    
    # Return top keywords
    return sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top_n]

def create_wordcloud(filtered_df, text_column='cleaned_text'):
    """Create word cloud from filtered tweets."""
    if len(filtered_df) == 0:
        return None
    
    # Combine all text
    all_text = ' '.join(filtered_df[text_column].fillna('').astype(str))
    
    if len(all_text.strip()) == 0:
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=400, 
        height=300, 
        background_color='white',
        max_words=50,
        colormap='viridis'
    ).generate(all_text)
    
    return wordcloud

def show_topic_analysis_page():
    """Display the professional analytics dashboard."""
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            color: #1f2937;
            text-align: center;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #6b7280;
            text-align: center;
            margin-bottom: 2rem;
            font-family: 'Inter', sans-serif;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            color: white;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .insight-box {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid #0ea5e9;
            margin: 1rem 0;
            color: white;
            font-weight: 500;
        }
        .search-container {
            background: #f8fafc;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 2px solid #e2e8f0;
        }
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        .tweet-container {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #3b82f6;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
            font-family: 'Inter', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Professional Header
    st.markdown('<h1 class="main-header">Social Media Trend Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze sentiment and trends for any keyword</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border: 2px solid #e5e7eb; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    # Load cached data
    app_data = st.session_state.get('app_data')
    if not app_data:
        st.warning("Please wait for data to load or refresh the page.")
        return
    
    df = app_data['df']
    model = app_data['model']
    feature_engineer = app_data['feature_engineer']
    
    # Professional Search Section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Center the search bar
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        keyword = st.text_input(
            "",
            placeholder="Enter keyword (e.g., iphone, IPL, movie)",
            key="search_keyword",
            label_visibility="collapsed"
        )
        
        col_search1, col_search2, col_search3 = st.columns([1, 2, 1])
        with col_search2:
            analyze_button = st.button("Analyze", type="primary", use_container_width=True, key="analyze_btn")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick keyword suggestions
    if not keyword:
        st.markdown("### Popular Keywords")
        popular_keywords = ["ipl", "iphone", "movie", "cricket", "love", "happy", "sad", "phone"]
        
        cols = st.columns(8)
        for i, kw in enumerate(popular_keywords[:8]):
            with cols[i]:
                if st.button(kw, key=f"suggestion_{kw}", use_container_width=True):
                    st.session_state.search_keyword = kw
                    st.rerun()
    
    if keyword and analyze_button:
        st.markdown("---")
        
        # Filter tweets by keyword
        filtered_df = filter_tweets_by_keyword(df, keyword)
        
        if len(filtered_df) == 0:
            st.warning(f"No tweets found containing keyword '{keyword}'")
            st.info("Try a different keyword or check spelling.")
            return
        
        # Professional Metrics Section
        st.markdown("### Analytics Overview")
        
        sentiment_results = analyze_keyword_sentiment(filtered_df, model, feature_engineer.vectorizer)
        
        # Card-style metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: bold;">{sentiment_results['total_count']}</div>
                <div style="font-size: 0.9rem;">Total Tweets</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <div style="font-size: 2rem; font-weight: bold;">{sentiment_results['positive_count']}</div>
                <div style="font-size: 0.9rem;">Positive Tweets</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <div style="font-size: 2rem; font-weight: bold;">{sentiment_results['negative_count']}</div>
                <div style="font-size: 0.9rem;">Negative Tweets</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sentiment_color = "#10b981" if sentiment_results['positive_percentage'] > 60 else "#f59e0b" if sentiment_results['positive_percentage'] > 40 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {sentiment_color} 0%, {sentiment_color}dd 100%);">
                <div style="font-size: 2rem; font-weight: bold;">{sentiment_results['positive_percentage']:.1f}%</div>
                <div style="font-size: 0.9rem;">Positive Sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphs Section
        st.markdown("### Visual Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Sentiment Distribution")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            sentiments = ['Positive', 'Negative']
            counts = [sentiment_results['positive_count'], sentiment_results['negative_count']]
            colors = ['#10b981', '#ef4444']
            
            bars = ax.bar(sentiments, counts, color=colors, alpha=0.8, width=0.6)
            ax.set_title(f'Sentiment Analysis for "{keyword}"', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Number of Tweets', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Trending Words")
            
            # Create word cloud
            wordcloud = create_wordcloud(filtered_df)
            if wordcloud:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Word Cloud for "{keyword}"', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No words available for word cloud")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Top Keywords Section
        st.markdown("### Trending Keywords")
        top_keywords = get_top_keywords_for_tweets(filtered_df)
        
        if top_keywords:
            # Create professional table
            keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
            keywords_df['Percentage'] = (keywords_df['Frequency'] / keywords_df['Frequency'].sum() * 100).round(1)
            
            # Style the table
            st.dataframe(
                keywords_df.head(10).style
                .background_gradient(subset=['Frequency'], cmap='Blues')
                .format({'Frequency': '{:,}', 'Percentage': '{:.1f}%'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trending keywords found.")
        
        # Sample Tweets Section
        st.markdown("### Sample Tweets")
        
        # Get positive and negative samples
        positive_tweets = filtered_df[filtered_df['sentiment'] == 1].head(3)
        negative_tweets = filtered_df[filtered_df['sentiment'] == 0].head(3)
        
        if len(positive_tweets) > 0:
            st.markdown("#### Positive Sentiment")
            for idx, row in positive_tweets.iterrows():
                st.markdown(f'''
                <div class="tweet-container">
                    <strong>Positive:</strong> {row['text']}
                </div>
                ''', unsafe_allow_html=True)
        
        if len(negative_tweets) > 0:
            st.markdown("#### Negative Sentiment")
            for idx, row in negative_tweets.iterrows():
                st.markdown(f'''
                <div class="tweet-container" style="border-left-color: #ef4444;">
                    <strong>Negative:</strong> {row['text']}
                </div>
                ''', unsafe_allow_html=True)
        
        # Insights Section
        st.markdown("### Key Insights")
        
        # Generate insights
        pos_pct = sentiment_results['positive_percentage']
        
        if pos_pct > 70:
            main_insight = f"**Very Positive Response**: '{keyword}' generates highly positive sentiment ({pos_pct:.1f}% positive)"
            color = "#10b981"
        elif pos_pct > 55:
            main_insight = f"**Generally Positive**: '{keyword}' is well-received with {pos_pct:.1f}% positive sentiment"
            color = "#3b82f6"
        elif pos_pct > 45:
            main_insight = f"**Mixed Reactions**: '{keyword}' shows divided opinions ({pos_pct:.1f}% positive)"
            color = "#f59e0b"
        else:
            main_insight = f"**Negative Trend**: '{keyword}' tends to generate negative responses ({pos_pct:.1f}% positive)"
            color = "#ef4444"
        
        # Display insights
        st.markdown(f'''
        <div class="insight-box" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            {main_insight}
        </div>
        ''', unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("#### Additional Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Volume Analysis**: Found {sentiment_results['total_count']:,} tweets mentioning '{keyword}'")
            st.info(f"**Dataset Share**: Represents {(len(filtered_df) / len(df) * 100):.2f}% of total dataset")
        
        with col2:
            if top_keywords:
                st.info(f"**Top Related Term**: '{top_keywords[0][0]}' (mentioned {top_keywords[0][1]} times)")
            avg_length = filtered_df['text'].str.len().mean()
            st.info(f"**Average Length**: {avg_length:.0f} characters per tweet")
    
    else:
        # Instructions when no keyword is entered
        st.markdown("---")
        st.markdown("### How to Use This Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Enter Keyword**
            - Type any keyword in the search box
            - Examples: iphone, IPL, movie, love, sad
            - Click "Analyze" to see results
            """)
        
        with col2:
            st.markdown("""
            **2. View Analytics**
            - Sentiment distribution charts
            - Trending keywords and word clouds
            - Professional metrics and insights
            """)
        
        with col3:
            st.markdown("""
            **3. Explore Insights**
            - Sample tweets with sentiment
            - Key trend analysis
            - Data-driven recommendations
            """)
        
        st.markdown("---")
        st.markdown("### Try These Keywords")
        
        example_keywords = [
            ("Technology", ["iphone", "phone", "app", "computer", "tech"]),
            ("Sports", ["cricket", "match", "game", "team", "player"]),
            ("Entertainment", ["movie", "film", "music", "show", "watch"]),
            ("Emotions", ["love", "happy", "sad", "hate", "excited"])
        ]
        
        for category, keywords in example_keywords:
            st.markdown(f"**{category}**: {', '.join(keywords)}")

# Run the page
if __name__ == "__main__":
    show_topic_analysis_page()
