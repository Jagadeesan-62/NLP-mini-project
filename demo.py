#!/usr/bin/env python3
"""
Demo script to show the application functionality
"""

import pandas as pd
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer
from model import SentimentModel
from topic_analyzer import TopicAnalyzer

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis functionality."""
    print("🎯 Sentiment Analysis Demo")
    print("=" * 50)
    
    # Sample tweets for testing
    test_tweets = [
        "I love this movie! It's absolutely amazing and fantastic!",
        "This phone is terrible. Worst purchase ever, very disappointed.",
        "Great cricket match today! The team played excellently.",
        "The new AI app is revolutionary and incredibly useful!",
        "Boring film, waste of time and money.",
        "Amazing technology! This will change everything."
    ]
    
    # Initialize components (with small sample for demo)
    preprocessor = TextPreprocessor()
    
    print("\n📝 Processing sample tweets:")
    for i, tweet in enumerate(test_tweets, 1):
        cleaned = preprocessor.preprocess_text(tweet)
        print(f"{i}. Original: {tweet}")
        print(f"   Cleaned:  {cleaned}")
        print()
    
    # Topic identification demo
    topic_analyzer = TopicAnalyzer()
    print("🏷️  Topic Identification:")
    for i, tweet in enumerate(test_tweets, 1):
        topic = topic_analyzer.identify_topic(tweet)
        print(f"{i}. '{tweet[:50]}...' -> {topic}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the full application:")
    print("streamlit run app.py")

if __name__ == "__main__":
    demo_sentiment_analysis()
