import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_comparison_page():
    """Display the topic comparison page."""
    
    st.markdown('<h1 class="main-header">⚖️ Topic Comparison</h1>', unsafe_allow_html=True)
    
    # Load cached data
    app_data = st.session_state.get('app_data')
    if not app_data:
        st.warning("Please wait for data to load or refresh the page.")
        return
    
    sentiment_results = app_data['sentiment_results']
    topic_trends = app_data['topic_trends']
    
    # Overall comparison chart
    st.markdown("### 📊 Sentiment Comparison Across Topics")
    fig = app_data['visualizer'].plot_topic_comparison(sentiment_results)
    st.pyplot(fig)
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Comparison")
    
    comparison_data = []
    for topic, data in sentiment_results.items():
        if data['total_tweets'] > 0:
            comparison_data.append({
                'Topic': topic,
                'Total Tweets': data['total_tweets'],
                'Positive Count': data['positive_count'],
                'Negative Count': data['negative_count'],
                'Positive %': f"{data['positive_percentage']:.1f}%",
                'Negative %': f"{data['negative_percentage']:.1f}%",
                'Sentiment Ratio': f"{data['positive_count']}/{data['negative_count']}" if data['negative_count'] > 0 else f"{data['positive_count']}/0"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Topic word clouds
    st.markdown("### ☁️ Topic Word Clouds")
    fig = app_data['visualizer'].create_topic_wordclouds(topic_trends)
    st.pyplot(fig)
    
    # Statistical analysis
    st.markdown("### 📈 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Performers")
        
        # Most positive topic
        most_positive = max(sentiment_results.items(), 
                          key=lambda x: x[1]['positive_percentage'] if x[1]['total_tweets'] > 0 else 0)
        
        # Most discussed topic
        most_discussed = max(sentiment_results.items(), 
                           key=lambda x: x[1]['total_tweets'])
        
        # Most balanced topic (closest to 50-50 split)
        most_balanced = min(sentiment_results.items(),
                          key=lambda x: abs(x[1]['positive_percentage'] - 50) if x[1]['total_tweets'] > 0 else 100)
        
        st.success(f"**Most Positive:** {most_positive[0]} ({most_positive[1]['positive_percentage']:.1f}% positive)")
        st.info(f"**Most Discussed:** {most_discussed[0]} ({most_discussed[1]['total_tweets']} tweets)")
        st.warning(f"**Most Balanced:** {most_balanced[0]} ({most_balanced[1]['positive_percentage']:.1f}% positive)")
    
    with col2:
        st.markdown("#### 📊 Engagement Metrics")
        
        # Calculate engagement metrics
        total_tweets = sum(data['total_tweets'] for data in sentiment_results.values())
        avg_tweets_per_topic = total_tweets / len([data for data in sentiment_results.values() if data['total_tweets'] > 0])
        
        # Positive sentiment rate
        total_positive = sum(data['positive_count'] for data in sentiment_results.values())
        overall_positive_rate = (total_positive / total_tweets) * 100 if total_tweets > 0 else 0
        
        st.metric("Total Topics Analyzed", len([data for data in sentiment_results.values() if data['total_tweets'] > 0]))
        st.metric("Avg Tweets per Topic", f"{avg_tweets_per_topic:.0f}")
        st.metric("Overall Positive Rate", f"{overall_positive_rate:.1f}%")
    
    # Sentiment distribution comparison
    st.markdown("### 🎯 Sentiment Distribution Comparison")
    
    # Create a comprehensive comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    topics = [topic for topic, data in sentiment_results.items() if data['total_tweets'] > 0]
    positive_counts = [sentiment_results[topic]['positive_count'] for topic in topics]
    negative_counts = [sentiment_results[topic]['negative_count'] for topic in topics]
    total_counts = [sentiment_results[topic]['total_tweets'] for topic in topics]
    positive_percentages = [sentiment_results[topic]['positive_percentage'] for topic in topics]
    
    # Chart 1: Tweet counts by topic
    ax1.bar(topics, total_counts, color='skyblue', alpha=0.8)
    ax1.set_title('Total Tweets by Topic', fontweight='bold')
    ax1.set_ylabel('Number of Tweets')
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Positive vs Negative comparison
    x = range(len(topics))
    width = 0.35
    ax2.bar([i - width/2 for i in x], positive_counts, width, label='Positive', color='green', alpha=0.8)
    ax2.bar([i + width/2 for i in x], negative_counts, width, label='Negative', color='red', alpha=0.8)
    ax2.set_title('Positive vs Negative Tweets', fontweight='bold')
    ax2.set_ylabel('Number of Tweets')
    ax2.set_xticks(x)
    ax2.set_xticklabels(topics, rotation=45)
    ax2.legend()
    
    # Chart 3: Positive percentage
    colors = ['green' if pct > 50 else 'orange' if pct > 40 else 'red' for pct in positive_percentages]
    ax3.bar(topics, positive_percentages, color=colors, alpha=0.8)
    ax3.set_title('Positive Sentiment Percentage', fontweight='bold')
    ax3.set_ylabel('Positive Sentiment (%)')
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Mark')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Chart 4: Topic engagement (total tweets vs sentiment)
    scatter = ax4.scatter(total_counts, positive_percentages, 
                        s=100, alpha=0.7, c=range(len(topics)), cmap='viridis')
    ax4.set_title('Engagement vs Sentiment', fontweight='bold')
    ax4.set_xlabel('Total Tweets')
    ax4.set_ylabel('Positive Sentiment (%)')
    
    # Add topic labels to scatter plot
    for i, topic in enumerate(topics):
        ax4.annotate(topic, (total_counts[i], positive_percentages[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Keyword comparison
    st.markdown("### 🔍 Top Keywords Comparison")
    
    # Create a comparison table for top keywords
    keyword_comparison = []
    for topic in topics:
        if topic_trends[topic]['top_keywords']:
            top_5_keywords = topic_trends[topic]['top_keywords'][:5]
            keywords_str = ', '.join([f"{kw[0]} ({kw[1]})" for kw in top_5_keywords])
            keyword_comparison.append({
                'Topic': topic,
                'Top 5 Keywords': keywords_str
            })
    
    if keyword_comparison:
        keywords_df = pd.DataFrame(keyword_comparison)
        st.dataframe(keywords_df, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 📊 Correlation Analysis")
    
    if len(topics) > 1:
        # Calculate correlations
        correlation_data = {
            'Topic': topics,
            'Tweet_Count': total_counts,
            'Positive_Percentage': positive_percentages,
            'Positive_Count': positive_counts,
            'Negative_Count': negative_counts
        }
        
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df[['Tweet_Count', 'Positive_Percentage', 'Positive_Count', 'Negative_Count']].corr()
        
        st.markdown("#### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        
        # Insights from correlations
        st.markdown("#### 🔍 Key Insights")
        tweet_sentiment_corr = correlation_matrix.loc['Tweet_Count', 'Positive_Percentage']
        
        if abs(tweet_sentiment_corr) > 0.5:
            if tweet_sentiment_corr > 0:
                st.info(f"**Strong Positive Correlation ({tweet_sentiment_corr:.2f}):** Topics with more tweets tend to have higher positive sentiment rates.")
            else:
                st.warning(f"**Strong Negative Correlation ({tweet_sentiment_corr:.2f}):** Topics with more tweets tend to have lower positive sentiment rates.")
        else:
            st.info(f"**Weak Correlation ({tweet_sentiment_corr:.2f}):** Tweet volume doesn't strongly correlate with sentiment positivity.")

if __name__ == "__main__":
    show_comparison_page()
