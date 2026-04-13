import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_insights_page():
    """Display the insights page."""
    
    st.markdown('<h1 class="main-header"> insights</h1>', unsafe_allow_html=True)
    
    # Load cached data
    app_data = st.session_state.get('app_data')
    if not app_data:
        st.warning("Please wait for data to load or refresh the page.")
        return
    
    sentiment_results = app_data['sentiment_results']
    topic_trends = app_data['topic_trends']
    insights = app_data['insights']
    
    # Display all insights
    st.markdown("### Key Findings")
    for i, insight in enumerate(insights, 1):
        st.markdown(f'<div class="insight-box">{i}. {insight}</div>', unsafe_allow_html=True)
    
    # Additional analysis
    st.markdown("### Additional Analysis")
    
    # Most positive topic
    most_positive = max(sentiment_results.items(), 
                      key=lambda x: x[1]['positive_percentage'] if x[1]['total_tweets'] > 0 else 0)
    
    # Most discussed topic
    most_discussed = max(sentiment_results.items(), 
                       key=lambda x: x[1]['total_tweets'])
    
    # Most negative topic
    most_negative = max(sentiment_results.items(), 
                       key=lambda x: x[1]['negative_percentage'] if x[1]['total_tweets'] > 0 else 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Most Positive Topic")
        st.success(f"**{most_positive[0]}** with {most_positive[1]['positive_percentage']:.1f}% positive sentiment")
    
    with col2:
        st.markdown("#### Most Discussed Topic")
        st.info(f"**{most_discussed[0]}** with {most_discussed[1]['total_tweets']} tweets")
    
    # Most negative topic
    if most_negative[1]['total_tweets'] > 0:
        st.markdown("#### Most Negative Topic")
        st.error(f"**{most_negative[0]}** with {most_negative[1]['negative_percentage']:.1f}% negative sentiment")
    
    # Overall sentiment summary
    total_positive = sum(data['positive_count'] for data in sentiment_results.values())
    total_negative = sum(data['negative_count'] for data in sentiment_results.values())
    total_tweets = total_positive + total_negative
    
    if total_tweets > 0:
        overall_positive = (total_positive / total_tweets) * 100
        
        st.markdown("### Overall Sentiment Summary")
        
        if overall_positive > 60:
            st.success(f"Overall sentiment is **POSITIVE** ({overall_positive:.1f}% positive)")
        elif overall_positive < 40:
            st.error(f"Overall sentiment is **NEGATIVE** ({overall_positive:.1f}% positive)")
        else:
            st.warning(f"Overall sentiment is **NEUTRAL** ({overall_positive:.1f}% positive)")
        
        # Progress bar
        st.progress(overall_positive / 100, text=f"Positive Sentiment: {overall_positive:.1f}%")
    
    # Detailed statistics
    st.markdown("### Detailed Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Topics", len([data for data in sentiment_results.values() if data['total_tweets'] > 0]))
        st.metric("Total Tweets", total_tweets)
        st.metric("Positive Tweets", total_positive)
    
    with col2:
        st.metric("Negative Tweets", total_negative)
        avg_positive = sum(data['positive_percentage'] for data in sentiment_results.values() if data['total_tweets'] > 0) / len([data for data in sentiment_results.values() if data['total_tweets'] > 0])
        st.metric("Avg Positive Rate", f"{avg_positive:.1f}%")
        
        # Most balanced topic
        most_balanced = min(sentiment_results.items(),
                          key=lambda x: abs(x[1]['positive_percentage'] - 50) if x[1]['total_tweets'] > 0 else 100)
        st.metric("Most Balanced", most_balanced[0])
    
    with col3:
        # Topic diversity
        topic_diversity = len([data for data in sentiment_results.values() if data['total_tweets'] > 0])
        st.metric("Active Topics", topic_diversity)
        
        # Engagement rate
        engagement_rate = total_tweets / topic_diversity if topic_diversity > 0 else 0
        st.metric("Avg Engagement", f"{engagement_rate:.0f} tweets/topic")
        
        # Sentiment variance
        positive_rates = [data['positive_percentage'] for data in sentiment_results.values() if data['total_tweets'] > 0]
        if len(positive_rates) > 1:
            sentiment_variance = pd.Series(positive_rates).var()
            st.metric("Sentiment Variance", f"{sentiment_variance:.2f}")
    
    # Topic performance radar chart
    st.markdown("### Topic Performance Radar")
    
    topics = [topic for topic, data in sentiment_results.items() if data['total_tweets'] > 0]
    if len(topics) > 0:
        # Create metrics for radar chart
        metrics = {
            'Tweet Volume': [sentiment_results[topic]['total_tweets'] for topic in topics],
            'Positive Rate': [sentiment_results[topic]['positive_percentage'] for topic in topics],
            'Engagement': [topic_trends[topic]['percentage'] for topic in topics],
            'Content Length': [topic_trends[topic]['avg_text_length'] for topic in topics]
        }
        
        # Normalize metrics to 0-100 scale
        normalized_metrics = {}
        for metric_name, values in metrics.items():
            max_val = max(values) if max(values) > 0 else 1
            normalized_metrics[metric_name] = [(v / max_val) * 100 for v in values]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set3(range(len(topics)))
        
        for i, topic in enumerate(topics):
            values = [normalized_metrics[metric][i] for metric in metrics.keys()]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=topic, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys())
        ax.set_ylim(0, 100)
        ax.set_title('Topic Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        st.pyplot(fig)
    
    # Trend analysis
    st.markdown("### Trend Analysis")
    
    # Identify trending topics based on engagement and sentiment
    trending_scores = {}
    for topic in topics:
        # Calculate trending score (combination of volume, sentiment, and engagement)
        volume_score = sentiment_results[topic]['total_tweets'] / max(total_tweets, 1) * 100
        sentiment_score = sentiment_results[topic]['positive_percentage']
        engagement_score = topic_trends[topic]['percentage']
        
        trending_scores[topic] = (volume_score + sentiment_score + engagement_score) / 3
    
    # Sort by trending score
    sorted_trending = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown("#### Trending Topics Ranking")
    for i, (topic, score) in enumerate(sorted_trending[:5], 1):
        st.write(f"{i}. **{topic}** - Trending Score: {score:.1f}/100")
    
    # Create trending score visualization
    if len(sorted_trending) > 0:
        trending_topics = [item[0] for item in sorted_trending]
        trending_scores_values = [item[1] for item in sorted_trending]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(trending_topics, trending_scores_values, color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trending Score', fontsize=12, fontweight='bold')
        ax.set_title('Topic Trending Scores', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig)
    
    # Recommendations
    st.markdown("### Recommendations")
    
    recommendations = []
    
    if most_positive[1]['positive_percentage'] > 70:
        recommendations.append(f"**{most_positive[0]}** shows very positive sentiment - consider leveraging this for marketing campaigns")
    
    if most_negative[1]['negative_percentage'] > 60:
        recommendations.append(f"**{most_negative[0]}** needs attention - investigate causes of negative sentiment and consider improvement strategies")
    
    if most_discussed[1]['total_tweets'] > total_tweets * 0.4:
        recommendations.append(f"**{most_discussed[0]}** dominates the conversation - ensure adequate resources for monitoring and engagement")
    
    # Content strategy recommendations
    if avg_positive > 65:
        recommendations.append("Overall sentiment is positive - maintain current content strategy and engagement approach")
    elif avg_positive < 45:
        recommendations.append("Overall sentiment needs improvement - focus on addressing negative feedback and enhancing positive content")
    else:
        recommendations.append("Sentiment is balanced - opportunity to strategically improve engagement in specific areas")
    
    # Topic-specific recommendations
    for topic in topics:
        if sentiment_results[topic]['positive_percentage'] > 75:
            recommendations.append(f"Expand **{topic}** content - high positive sentiment indicates strong audience approval")
        elif sentiment_results[topic]['negative_percentage'] > 60:
            recommendations.append(f"Address issues in **{topic}** - high negative sentiment requires immediate attention")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Action items
    st.markdown("### Action Items")
    st.markdown("""
    #### Immediate Actions:
    - Monitor trending topics daily to stay ahead of public sentiment
    - Set up alerts for significant sentiment changes
    - Engage with high-performing topics to maintain positive momentum
    
    #### Strategic Planning:
    - Use sentiment insights to guide content strategy
    - Allocate resources based on topic performance
    - Develop contingency plans for negative sentiment spikes
    
    #### Continuous Improvement:
    - Regularly analyze sentiment patterns and trends
    - A/B test content strategies for different topics
    - Gather user feedback to refine analysis approach
    """)

if __name__ == "__main__":
    show_insights_page()
