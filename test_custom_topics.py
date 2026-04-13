#!/usr/bin/env python3
"""
Test custom topics functionality
"""

import pandas as pd
from data_handler import DataHandler
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model import SentimentModel
from topic_analyzer import TopicAnalyzer

def test_custom_topics():
    """Test custom topics functionality."""
    print("🧪 Testing Custom Topics Functionality")
    print("=" * 50)
    
    # Initialize components
    data_handler = DataHandler()
    preprocessor = TextPreprocessor()
    feature_engineer = FeatureEngineer()
    model = SentimentModel()
    topic_analyzer = TopicAnalyzer()
    
    # Load data
    print("Loading data...")
    df = data_handler.load_data(sample_size=1000)
    df_processed = preprocessor.preprocess_dataframe(df, 'text', 'cleaned_text')
    
    # Test default topics
    print("\n🔍 Testing default topics...")
    sample_texts = [
        "I love watching movies and films",
        "The cricket match was amazing",
        "My new phone has great AI features",
        "This is about politics and government",
        "Weather today is very nice"
    ]
    
    for text in sample_texts:
        topic = topic_analyzer.identify_topic(text)
        print(f"'{text}' -> {topic}")
    
    # Test custom topics
    print("\n🏷️ Testing custom topics...")
    custom_topics = {
        'Politics': ['politics', 'government', 'election', 'vote', 'president', 'policy'],
        'Weather': ['weather', 'rain', 'sunny', 'temperature', 'climate', 'storm'],
        'Food': ['food', 'restaurant', 'cooking', 'recipe', 'delicious', 'meal']
    }
    
    # Update topic analyzer
    topic_analyzer.topic_keywords = custom_topics
    
    print("Updated topics:", list(custom_topics.keys()))
    
    for text in sample_texts:
        topic = topic_analyzer.identify_topic(text)
        print(f"'{text}' -> {topic}")
    
    # Train model first
    print("\n🤖 Training model...")
    X = feature_engineer.fit_transform(df_processed['cleaned_text'])
    y = df_processed['sentiment']
    X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_split(X, y)
    model.train(X_train, y_train)
    
    # Test topic sentiment analysis
    print("\n📊 Testing topic sentiment analysis...")
    sentiment_results = topic_analyzer.get_topic_sentiment_analysis(
        df_processed, model, feature_engineer.vectorizer, 'text'
    )
    
    print("Available topics for analysis:", list(sentiment_results.keys()))
    for topic, results in sentiment_results.items():
        print(f"{topic}: {results['positive_count']} positive, {results['negative_count']} negative")
    
    print("\n✅ Custom topics test completed successfully!")

if __name__ == "__main__":
    test_custom_topics()
