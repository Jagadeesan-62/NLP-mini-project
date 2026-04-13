from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
from io import BytesIO
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handler import DataHandler
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model import SentimentModel
from topic_analyzer import TopicAnalyzer
from visualization import Visualizer

app = Flask(__name__)

# Global variables for cached data
cached_data = None
cached_models = None

def load_data_and_models():
    """Load data and train models (cached)."""
    global cached_data, cached_models
    
    if cached_data is not None and cached_models is not None:
        return cached_data, cached_models
    
    try:
        # Initialize components
        data_handler = DataHandler()
        preprocessor = TextPreprocessor()
        feature_engineer = FeatureEngineer()
        model = SentimentModel()
        topic_analyzer = TopicAnalyzer()
        visualizer = Visualizer()
        
        # Load data
        data_path = "data/sentiment140.csv"
        if not os.path.exists(data_path):
            data_path = "sentiment140.csv"
        
        if not os.path.exists(data_path):
            return None, None
        
        df = data_handler.load_data(sample_size=50000)
        df_processed = preprocessor.preprocess_dataframe(df, 'text', 'cleaned_text')
        
        # Feature engineering
        X = feature_engineer.fit_transform(df_processed['cleaned_text'])
        y = df_processed['sentiment']
        
        # Train model
        X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_split(X, y)
        model.train(X_train, y_train)
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test)
        
        # Cache the results
        cached_data = {
            'df': df_processed,
            'evaluation': evaluation
        }
        
        cached_models = {
            'preprocessor': preprocessor,
            'feature_engineer': feature_engineer,
            'model': model,
            'topic_analyzer': topic_analyzer,
            'visualizer': visualizer
        }
        
        return cached_data, cached_models
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def filter_tweets_by_keyword(df, keyword):
    """Filter tweets containing the keyword."""
    if not keyword or not isinstance(keyword, str):
        return pd.DataFrame()
    
    keyword = keyword.strip().lower()
    pattern = r'\b' + re.escape(keyword) + r'\b'
    mask = df['text'].str.lower().str.contains(pattern, na=False, regex=True)
    return df[mask].copy()

def analyze_keyword_sentiment(filtered_df, model, vectorizer):
    """Analyze sentiment for filtered tweets."""
    if len(filtered_df) == 0:
        return {
            'total_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'positive_percentage': 0.0,
            'negative_percentage': 0.0
        }
    
    cleaned_texts = filtered_df['cleaned_text'].fillna('')
    text_vectors = vectorizer.transform(cleaned_texts)
    predictions = model.predict(text_vectors)
    
    positive_count = int(sum(predictions == 1))
    negative_count = int(sum(predictions == 0))
    total_count = int(len(filtered_df))
    
    return {
        'total_count': total_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': float((positive_count / total_count) * 100),
        'negative_percentage': float((negative_count / total_count) * 100)
    }

def get_top_keywords_for_tweets(filtered_df, top_n=10):
    """Extract top keywords from filtered tweets."""
    if len(filtered_df) == 0:
        return []
    
    all_text = ' '.join(filtered_df['cleaned_text'].fillna('').astype(str))
    words = all_text.lower().split()
    word_freq = Counter(words)
    
    stopwords = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'i', 'it', 'for', 'of', 'in', 'this', 'that', 'was', 'with', 'as', 'be', 'are', 'have', 'had', 'but', 'not', 'you', 'was', 'were', 'he', 'she', 'they', 'we', 'so', 'if', 'or', 'by', 'an', 'will', 'my', 'your', 'can', 'has', 'been', 'would', 'there', 'their', 'what', 'when', 'who', 'how', 'all', 'about', 'up', 'out', 'do', 'them', 'then', 'than', 'some', 'time', 'very', 'just', 'get', 'go', 'me', 'no', 'like', 'know', 'make', 'take', 'see', 'come', 'think', 'back', 'well', 'only', 'even', 'way', 'good', 'new', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'after', 'work', 'life', 'here', 'own', 'without', 'again', 'off', 'old', 'while', 'still', 'over', 'between', 'many', 'before', 'through', 'being', 'much', 'where', 'too', 'more'}
    
    filtered_words = [(word, int(count)) for word, count in word_freq.items() 
                      if len(word) > 2 and word not in stopwords and not word.isdigit()]
    
    return sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top_n]

def create_sentiment_chart(sentiment_data, keyword):
    """Create sentiment distribution chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiments = ['Positive', 'Negative']
    counts = [sentiment_data['positive_count'], sentiment_data['negative_count']]
    colors = ['#10b981', '#ef4444']
    
    bars = ax.bar(sentiments, counts, color=colors, alpha=0.8, width=0.6)
    ax.set_title(f'Sentiment Analysis for "{keyword}"', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Tweets', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def create_wordcloud(filtered_df):
    """Create word cloud from filtered tweets."""
    if len(filtered_df) == 0:
        return None
    
    all_text = ' '.join(filtered_df['cleaned_text'].fillna('').astype(str))
    
    if len(all_text.strip()) == 0:
        return None
    
    wordcloud = WordCloud(
        width=400, 
        height=300, 
        background_color='white',
        max_words=50,
        colormap='viridis'
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def create_comparison_chart(keywords_data):
    """Create comparison chart for multiple keywords."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    keywords = list(keywords_data.keys())
    positive_percentages = [data['positive_percentage'] for data in keywords_data.values()]
    
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ef4444']
    bars = ax.bar(keywords, positive_percentages, color=colors[:len(keywords)], alpha=0.8)
    
    ax.set_title('Sentiment Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Positive Sentiment %', fontsize=12)
    ax.set_xlabel('Keywords', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def home():
    """Home page."""
    return render_template('home.html')

@app.route('/dataset')
def dataset():
    """Dataset overview page."""
    data, models = load_data_and_models()
    
    if data is None:
        return render_template('error.html', message="Dataset not found")
    
    df = data['df']
    evaluation = data['evaluation']
    
    # Dataset info
    dataset_info = {
        'total_rows': int(len(df)),
        'columns': int(len(df.columns)),
        'positive_tweets': int(len(df[df['sentiment'] == 1])),
        'negative_tweets': int(len(df[df['sentiment'] == 0]))
    }
    
    # Sample data
    sample_data = df.head(10)[['text', 'sentiment']].to_dict('records')
    
    # Convert sample data sentiment to regular int
    for record in sample_data:
        record['sentiment'] = int(record['sentiment'])
    
    # Sentiment distribution chart
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Positive', 'Negative']
    sizes = [sentiment_counts.get(1, 0), sentiment_counts.get(0, 0)]
    colors = ['#10b981', '#ef4444']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Convert evaluation to JSON-serializable format
    evaluation_serializable = {
        'accuracy': float(evaluation.accuracy) if hasattr(evaluation, 'accuracy') else 0.0,
        'precision': float(evaluation.precision) if hasattr(evaluation, 'precision') else 0.0,
        'recall': float(evaluation.recall) if hasattr(evaluation, 'recall') else 0.0,
        'f1_score': float(evaluation.f1_score) if hasattr(evaluation, 'f1_score') else 0.0
    }
    
    return render_template('dataset.html', 
                        dataset_info=dataset_info,
                        sample_data=sample_data,
                        chart_base64=chart_base64,
                        evaluation=evaluation_serializable)

@app.route('/search')
def search():
    """Search analysis page."""
    return render_template('search.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze keyword endpoint."""
    keyword = request.form.get('keyword', '').strip()
    
    if not keyword:
        return jsonify({'error': 'Please enter a keyword'})
    
    data, models = load_data_and_models()
    
    if data is None or models is None:
        return jsonify({'error': 'Models not loaded'})
    
    df = data['df']
    model = models['model']
    vectorizer = models['feature_engineer'].vectorizer
    
    # Filter tweets
    filtered_df = filter_tweets_by_keyword(df, keyword)
    
    if len(filtered_df) == 0:
        return jsonify({'error': f'No tweets found for keyword "{keyword}"'})
    
    # Analyze sentiment
    sentiment_data = analyze_keyword_sentiment(filtered_df, model, vectorizer)
    
    # Get top keywords
    top_keywords = get_top_keywords_for_tweets(filtered_df)
    
    # Create charts
    sentiment_chart = create_sentiment_chart(sentiment_data, keyword)
    wordcloud_chart = create_wordcloud(filtered_df)
    
    # Get sample tweets
    positive_tweets = filtered_df[filtered_df['sentiment'] == 1].head(3)['text'].tolist()
    negative_tweets = filtered_df[filtered_df['sentiment'] == 0].head(3)['text'].tolist()
    
    # Generate insight
    pos_pct = sentiment_data['positive_percentage']
    if pos_pct > 70:
        insight = f"Very Positive Response: '{keyword}' generates highly positive sentiment ({pos_pct:.1f}% positive)"
        color = "#10b981"
    elif pos_pct > 55:
        insight = f"Generally Positive: '{keyword}' is well-received with {pos_pct:.1f}% positive sentiment"
        color = "#3b82f6"
    elif pos_pct > 45:
        insight = f"Mixed Reactions: '{keyword}' shows divided opinions ({pos_pct:.1f}% positive)"
        color = "#f59e0b"
    else:
        insight = f"Negative Trend: '{keyword}' tends to generate negative responses ({pos_pct:.1f}% positive)"
        color = "#ef4444"
    
    return jsonify({
        'keyword': keyword,
        'sentiment_data': sentiment_data,
        'top_keywords': top_keywords[:10],
        'sentiment_chart': sentiment_chart,
        'wordcloud_chart': wordcloud_chart,
        'positive_tweets': positive_tweets,
        'negative_tweets': negative_tweets,
        'insight': insight,
        'insight_color': color
    })

@app.route('/comparison')
def comparison():
    """Comparison analysis page."""
    return render_template('comparison.html')

@app.route('/compare', methods=['POST'])
def compare():
    """Compare multiple keywords endpoint."""
    keywords = request.form.getlist('keywords')
    keywords = [k.strip() for k in keywords if k.strip()]
    
    if len(keywords) < 2:
        return jsonify({'error': 'Please enter at least 2 keywords to compare'})
    
    data, models = load_data_and_models()
    
    if data is None or models is None:
        return jsonify({'error': 'Models not loaded'})
    
    df = data['df']
    model = models['model']
    vectorizer = models['feature_engineer'].vectorizer
    
    keywords_data = {}
    
    for keyword in keywords:
        filtered_df = filter_tweets_by_keyword(df, keyword)
        
        if len(filtered_df) == 0:
            keywords_data[keyword] = {
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'positive_percentage': 0.0,
                'negative_percentage': 0.0
            }
        else:
            sentiment_data = analyze_keyword_sentiment(filtered_df, model, vectorizer)
            keywords_data[keyword] = sentiment_data
    
    # Create comparison chart
    comparison_chart = create_comparison_chart(keywords_data)
    
    # Generate insights
    sorted_keywords = sorted(keywords_data.items(), key=lambda x: x[1]['positive_percentage'], reverse=True)
    most_positive = sorted_keywords[0][0] if sorted_keywords else None
    most_negative = sorted_keywords[-1][0] if sorted_keywords else None
    
    insights = []
    if most_positive:
        insights.append(f"'{most_positive}' has the highest positive sentiment")
    if most_negative and most_negative != most_positive:
        insights.append(f"'{most_negative}' shows more negative responses")
    
    return jsonify({
        'keywords_data': keywords_data,
        'comparison_chart': comparison_chart,
        'insights': insights
    })

@app.route('/insights')
def insights():
    """Insights page."""
    return render_template('insights.html')

@app.route('/generate_insights', methods=['POST'])
def generate_insights():
    """Generate insights endpoint."""
    keywords = request.form.getlist('keywords')
    keywords = [k.strip() for k in keywords if k.strip()]
    
    if not keywords:
        return jsonify({'error': 'Please provide keywords for analysis'})
    
    data, models = load_data_and_models()
    
    if data is None or models is None:
        return jsonify({'error': 'Models not loaded'})
    
    df = data['df']
    model = models['model']
    vectorizer = models['feature_engineer'].vectorizer
    
    # Analyze each keyword
    keywords_analysis = {}
    for keyword in keywords:
        filtered_df = filter_tweets_by_keyword(df, keyword)
        if len(filtered_df) > 0:
            sentiment_data = analyze_keyword_sentiment(filtered_df, model, vectorizer)
            keywords_analysis[keyword] = sentiment_data
    
    # Generate summary insights
    if not keywords_analysis:
        return jsonify({'error': 'No data found for provided keywords'})
    
    # Find most positive, negative, and discussed
    most_positive = max(keywords_analysis.items(), key=lambda x: x[1]['positive_percentage'])
    most_negative = min(keywords_analysis.items(), key=lambda x: x[1]['positive_percentage'])
    most_discussed = max(keywords_analysis.items(), key=lambda x: x[1]['total_count'])
    
    # Generate observations
    observations = []
    
    # Overall sentiment patterns
    avg_positive = sum(data['positive_percentage'] for data in keywords_analysis.values()) / len(keywords_analysis)
    if avg_positive > 60:
        observations.append("Overall positive sentiment trend across all keywords")
    elif avg_positive < 40:
        observations.append("Overall negative sentiment trend across all keywords")
    else:
        observations.append("Mixed sentiment patterns across keywords")
    
    # High engagement topics
    high_engagement = [kw for kw, data in keywords_analysis.items() if data['total_count'] > 1000]
    if high_engagement:
        observations.append(f"High engagement topics: {', '.join(high_engagement)}")
    
    # Sentiment extremes
    if most_positive[1]['positive_percentage'] > 80:
        observations.append(f"Extremely positive sentiment for '{most_positive[0]}'")
    
    if most_negative[1]['positive_percentage'] < 20:
        observations.append(f"Extremely negative sentiment for '{most_negative[0]}'")
    
    # Generate conclusion
    if avg_positive > 60:
        conclusion = "Overall, social media shows positive trends for the analyzed topics."
    elif avg_positive < 40:
        conclusion = "Overall, social media shows negative trends for the analyzed topics."
    else:
        conclusion = "Overall, social media shows mixed responses for the analyzed topics."
    
    return jsonify({
        'most_positive': most_positive,
        'most_negative': most_negative,
        'most_discussed': most_discussed,
        'observations': observations,
        'conclusion': conclusion,
        'keywords_analysis': keywords_analysis
    })

if __name__ == '__main__':
    # Load data on startup
    print("Loading data and models...")
    load_data_and_models()
    print("Application ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
