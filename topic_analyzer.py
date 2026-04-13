import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
import re
from model import SentimentModel
from preprocessing import TextPreprocessor

class TopicAnalyzer:
    """Handles topic identification and trend analysis."""
    
    def __init__(self):
        self.topic_keywords = {
            'Movies': ['movie', 'film', 'actor', 'actress', 'director', 'cinema', 'theater', 'hollywood', 'watch', 'screen'],
            'Sports': ['cricket', 'match', 'team', 'game', 'player', 'sport', 'football', 'basketball', 'tennis', 'ipl', 'world cup'],
            'Technology': ['phone', 'mobile', 'ai', 'app', 'software', 'computer', 'laptop', 'tech', 'digital', 'internet', 'device']
        }
        self.preprocessor = TextPreprocessor()
    
    def identify_topic(self, text: str) -> str:
        """
        Identify the topic of a given text based on keywords.
        
        Args:
            text: Input text
            
        Returns:
            Identified topic or 'Other'
        """
        if not isinstance(text, str):
            return 'Other'
        
        # Clean and lowercase text
        cleaned_text = self.preprocessor.clean_text(text).lower()
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, cleaned_text))
                score += matches
            topic_scores[topic] = score
        
        # Return topic with highest score, or 'Other' if no matches
        if all(score == 0 for score in topic_scores.values()):
            return 'Other'
        
        return max(topic_scores, key=topic_scores.get)
    
    def filter_by_topic(self, df: pd.DataFrame, topic: str, 
                       text_column: str = 'text') -> pd.DataFrame:
        """
        Filter DataFrame by specific topic.
        
        Args:
            df: Input DataFrame
            topic: Topic to filter by
            text_column: Name of the text column
            
        Returns:
            Filtered DataFrame
        """
        if topic not in self.topic_keywords and topic != 'Other':
            raise ValueError(f"Unknown topic: {topic}")
        
        # Identify topic for each text
        df_copy = df.copy()
        df_copy['topic'] = df_copy[text_column].apply(self.identify_topic)
        
        # Filter by topic
        if topic == 'Other':
            filtered_df = df_copy[df_copy['topic'] == 'Other']
        else:
            filtered_df = df_copy[df_copy['topic'] == topic]
        
        return filtered_df.reset_index(drop=True)
    
    def get_topic_distribution(self, df: pd.DataFrame, 
                              text_column: str = 'text') -> Dict[str, int]:
        """
        Get distribution of topics in the dataset.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            Dictionary with topic counts
        """
        df_copy = df.copy()
        df_copy['topic'] = df_copy[text_column].apply(self.identify_topic)
        
        topic_counts = df_copy['topic'].value_counts().to_dict()
        
        # Ensure all topics are included
        for topic in list(self.topic_keywords.keys()) + ['Other']:
            if topic not in topic_counts:
                topic_counts[topic] = 0
        
        return topic_counts
    
    def get_top_keywords(self, df: pd.DataFrame, topic: str, 
                         text_column: str = 'cleaned_text', 
                         top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top keywords for a specific topic.
        
        Args:
            df: Input DataFrame
            topic: Topic to analyze
            text_column: Name of the cleaned text column
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        # Filter by topic
        topic_df = self.filter_by_topic(df, topic, text_column)
        
        if len(topic_df) == 0:
            return []
        
        # Combine all texts for the topic
        all_text = ' '.join(topic_df[text_column].astype(str))
        
        # Tokenize and count word frequencies
        words = all_text.split()
        word_freq = Counter(words)
        
        # Remove topic keywords (they're expected to be frequent)
        if topic in self.topic_keywords:
            for keyword in self.topic_keywords[topic]:
                word_freq.pop(keyword, None)
        
        # Get top keywords
        top_keywords = word_freq.most_common(top_n)
        
        return top_keywords
    
    def analyze_topic_trends(self, df: pd.DataFrame, 
                           text_column: str = 'cleaned_text') -> Dict[str, Dict[str, Any]]:
        """
        Analyze trends for all topics.
        
        Args:
            df: Input DataFrame
            text_column: Name of the cleaned text column
            
        Returns:
            Dictionary with trend analysis for each topic
        """
        trends = {}
        
        for topic in list(self.topic_keywords.keys()) + ['Other']:
            topic_df = self.filter_by_topic(df, topic, text_column)
            
            if len(topic_df) > 0:
                # Get top keywords
                top_keywords = self.get_top_keywords(df, topic, text_column, top_n=10)
                
                # Calculate basic statistics
                avg_text_length = topic_df[text_column].astype(str).str.len().mean()
                
                trends[topic] = {
                    'count': len(topic_df),
                    'percentage': (len(topic_df) / len(df)) * 100,
                    'top_keywords': top_keywords,
                    'avg_text_length': avg_text_length
                }
            else:
                trends[topic] = {
                    'count': 0,
                    'percentage': 0,
                    'top_keywords': [],
                    'avg_text_length': 0
                }
        
        return trends
    
    def get_topic_sentiment_analysis(self, df: pd.DataFrame, 
                                   model: SentimentModel,
                                   vectorizer,
                                   text_column: str = 'text') -> Dict[str, Dict[str, Any]]:
        """
        Perform sentiment analysis for each topic.
        
        Args:
            df: Input DataFrame
            model: Trained sentiment model
            vectorizer: Fitted TF-IDF vectorizer
            text_column: Name of the text column
            
        Returns:
            Dictionary with sentiment analysis for each topic
        """
        sentiment_results = {}
        
        for topic in list(self.topic_keywords.keys()) + ['Other']:
            topic_df = self.filter_by_topic(df, topic, text_column)
            
            if len(topic_df) > 0:
                # Preprocess texts
                cleaned_texts = topic_df[text_column].apply(self.preprocessor.preprocess_text)
                
                # Make predictions
                if len(cleaned_texts) > 0:
                    # Transform texts
                    text_vectors = vectorizer.transform(cleaned_texts)
                    predictions = model.predict(text_vectors)
                    
                    # Calculate sentiment distribution
                    positive_count = np.sum(predictions == 1)
                    negative_count = np.sum(predictions == 0)
                    total = len(predictions)
                    
                    sentiment_results[topic] = {
                        'total_tweets': total,
                        'positive_count': positive_count,
                        'negative_count': negative_count,
                        'positive_percentage': (positive_count / total) * 100,
                        'negative_percentage': (negative_count / total) * 100
                    }
                else:
                    sentiment_results[topic] = {
                        'total_tweets': 0,
                        'positive_count': 0,
                        'negative_count': 0,
                        'positive_percentage': 0,
                        'negative_percentage': 0
                    }
            else:
                sentiment_results[topic] = {
                    'total_tweets': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'positive_percentage': 0,
                    'negative_percentage': 0
                }
        
        return sentiment_results
    
    def generate_insights(self, sentiment_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate insights from sentiment analysis results.
        
        Args:
            sentiment_results: Results from topic sentiment analysis
            
        Returns:
            List of insights
        """
        insights = []
        
        # Find topic with highest positive sentiment
        max_positive_topic = max(sentiment_results.items(), 
                               key=lambda x: x[1]['positive_percentage'] if x[1]['total_tweets'] > 0 else 0)
        
        if max_positive_topic[1]['total_tweets'] > 0:
            insights.append(f"**{max_positive_topic[0]}** has the highest positive sentiment at {max_positive_topic[1]['positive_percentage']:.1f}%")
        
        # Find topic with highest negative sentiment
        max_negative_topic = max(sentiment_results.items(), 
                               key=lambda x: x[1]['negative_percentage'] if x[1]['total_tweets'] > 0 else 0)
        
        if max_negative_topic[1]['total_tweets'] > 0:
            insights.append(f"**{max_negative_topic[0]}** shows the most negative feedback at {max_negative_topic[1]['negative_percentage']:.1f}%")
        
        # Find most discussed topic
        most_discussed = max(sentiment_results.items(), 
                           key=lambda x: x[1]['total_tweets'])
        
        if most_discussed[1]['total_tweets'] > 0:
            insights.append(f"**{most_discussed[0]}** is the most discussed topic with {most_discussed[1]['total_tweets']} tweets")
        
        # Compare positive vs negative overall
        total_positive = sum(result['positive_count'] for result in sentiment_results.values())
        total_negative = sum(result['negative_count'] for result in sentiment_results.values())
        total_tweets = total_positive + total_negative
        
        if total_tweets > 0:
            overall_positive = (total_positive / total_tweets) * 100
            if overall_positive > 60:
                insights.append(f"Overall sentiment is **positive** ({overall_positive:.1f}% positive)")
            elif overall_positive < 40:
                insights.append(f"Overall sentiment is **negative** ({overall_positive:.1f}% positive)")
            else:
                insights.append(f"Overall sentiment is **neutral** ({overall_positive:.1f}% positive)")
        
        return insights

if __name__ == "__main__":
    # Test the topic analyzer
    sample_texts = [
        "I love this movie! The actor was amazing.",
        "Great cricket match today! The team played well.",
        "This new phone is fantastic! Amazing technology.",
        "The film was boring and disappointing.",
        "Cricket world cup is exciting! Best sport ever.",
        "AI technology is changing the world rapidly.",
        "Terrible movie, waste of time and money.",
        "Mobile phone battery life is terrible."
    ]
    
    sample_df = pd.DataFrame({'text': sample_texts})
    
    analyzer = TopicAnalyzer()
    
    print("Topic identification:")
    for text in sample_texts:
        topic = analyzer.identify_topic(text)
        print(f"'{text}' -> {topic}")
    
    print("\nTopic distribution:")
    distribution = analyzer.get_topic_distribution(sample_df)
    print(distribution)
    
    print("\nTopic trends:")
    trends = analyzer.analyze_topic_trends(sample_df, 'text')
    for topic, trend in trends.items():
        print(f"{topic}: {trend['count']} tweets ({trend['percentage']:.1f}%)")
