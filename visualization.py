import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Any
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """Handles visualization for sentiment analysis results."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_sentiment_distribution(self, sentiment_data: Dict[str, Dict[str, Any]], 
                                   save_path: str = None) -> plt.Figure:
        """
        Plot sentiment distribution for different topics.
        
        Args:
            sentiment_data: Dictionary with sentiment results for each topic
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        topics = []
        positive_percentages = []
        negative_percentages = []
        
        for topic, data in sentiment_data.items():
            if data['total_tweets'] > 0:
                topics.append(topic)
                positive_percentages.append(data['positive_percentage'])
                negative_percentages.append(data['negative_percentage'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(topics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, positive_percentages, width, 
                      label='Positive', alpha=0.8, color='green')
        bars2 = ax.bar(x + width/2, negative_percentages, width, 
                      label='Negative', alpha=0.8, color='red')
        
        ax.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution by Topic', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_topic_comparison(self, sentiment_data: Dict[str, Dict[str, Any]], 
                            save_path: str = None) -> plt.Figure:
        """
        Plot comparison of topics based on tweet counts and sentiment.
        
        Args:
            sentiment_data: Dictionary with sentiment results for each topic
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        topics = []
        tweet_counts = []
        positive_percentages = []
        
        for topic, data in sentiment_data.items():
            if data['total_tweets'] > 0:
                topics.append(topic)
                tweet_counts.append(data['total_tweets'])
                positive_percentages.append(data['positive_percentage'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Tweet counts
        bars1 = ax1.bar(topics, tweet_counts, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
        ax1.set_title('Tweet Count by Topic', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Positive sentiment percentages
        colors = ['green' if pct > 50 else 'orange' if pct > 40 else 'red' 
                 for pct in positive_percentages]
        bars2 = ax2.bar(topics, positive_percentages, color=colors, alpha=0.8)
        ax2.set_xlabel('Topics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Positive Sentiment (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Positive Sentiment by Topic', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Mark')
        ax2.legend()
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_wordcloud(self, text_data: str, title: str = "Word Cloud", 
                        save_path: str = None) -> plt.Figure:
        """
        Create a word cloud from text data.
        
        Args:
            text_data: Text data for word cloud
            title: Title for the word cloud
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            contour_width=2,
            contour_color='steelblue'
        ).generate(text_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_top_keywords(self, keywords: List[Tuple[str, int]], 
                         topic: str = "Topic", 
                         save_path: str = None) -> plt.Figure:
        """
        Plot top keywords for a topic.
        
        Args:
            keywords: List of (keyword, frequency) tuples
            topic: Topic name
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not keywords:
            # Create empty plot if no keywords
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No keywords found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Top Keywords for {topic}', fontsize=14, fontweight='bold')
            return fig
        
        # Prepare data
        words = [kw[0] for kw in keywords]
        frequencies = [kw[1] for kw in keywords]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(words, frequencies, color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Keywords', fontsize=12, fontweight='bold')
        ax.set_title(f'Top Keywords for {topic}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_topic_wordclouds(self, topic_trends: Dict[str, Dict[str, Any]], 
                               save_path: str = None) -> plt.Figure:
        """
        Create word clouds for multiple topics.
        
        Args:
            topic_trends: Dictionary with trend analysis for each topic
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter topics with data
        topics_with_data = {topic: data for topic, data in topic_trends.items() 
                          if data['top_keywords'] and topic != 'Other'}
        
        if not topics_with_data:
            # Create empty plot if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No topic data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            return fig
        
        # Calculate subplot grid
        n_topics = len(topics_with_data)
        cols = 2
        rows = (n_topics + 1) // 2
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_topics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create word cloud for each topic
        for idx, (topic, data) in enumerate(topics_with_data.items()):
            row = idx // cols
            col = idx % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Create text from keywords
            keywords_text = ' '.join([f"{kw[0]} " * kw[1] for kw in data['top_keywords']])
            
            # Create word cloud
            wordcloud = WordCloud(
                width=400, 
                height=300, 
                background_color='white',
                max_words=50,
                colormap='Set2'
            ).generate(keywords_text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{topic} ({data["count"]} tweets)', fontsize=12, fontweight='bold')
        
        # Hide empty subplots
        for idx in range(n_topics, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, sentiment_data: Dict[str, Dict[str, Any]], 
                      topic_trends: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Save all plots and return file paths.
        
        Args:
            sentiment_data: Sentiment analysis results
            topic_trends: Topic trend analysis results
            
        Returns:
            Dictionary with plot file paths
        """
        plot_paths = {}
        
        # Sentiment distribution
        sentiment_path = os.path.join(self.output_dir, 'sentiment_distribution.png')
        self.plot_sentiment_distribution(sentiment_data, sentiment_path)
        plot_paths['sentiment_distribution'] = sentiment_path
        
        # Topic comparison
        comparison_path = os.path.join(self.output_dir, 'topic_comparison.png')
        self.plot_topic_comparison(sentiment_data, comparison_path)
        plot_paths['topic_comparison'] = comparison_path
        
        # Topic word clouds
        wordcloud_path = os.path.join(self.output_dir, 'topic_wordclouds.png')
        self.create_topic_wordclouds(topic_trends, wordcloud_path)
        plot_paths['topic_wordclouds'] = wordcloud_path
        
        print(f"All plots saved to {self.output_dir} directory")
        
        return plot_paths

if __name__ == "__main__":
    # Test the visualizer
    sample_sentiment_data = {
        'Movies': {
            'total_tweets': 100,
            'positive_count': 60,
            'negative_count': 40,
            'positive_percentage': 60.0,
            'negative_percentage': 40.0
        },
        'Sports': {
            'total_tweets': 150,
            'positive_count': 90,
            'negative_count': 60,
            'positive_percentage': 60.0,
            'negative_percentage': 40.0
        },
        'Technology': {
            'total_tweets': 80,
            'positive_count': 32,
            'negative_count': 48,
            'positive_percentage': 40.0,
            'negative_percentage': 60.0
        }
    }
    
    sample_topic_trends = {
        'Movies': {
            'count': 100,
            'top_keywords': [('great', 20), ('amazing', 15), ('love', 12)]
        },
        'Sports': {
            'count': 150,
            'top_keywords': [('match', 30), ('team', 25), ('game', 20)]
        },
        'Technology': {
            'count': 80,
            'top_keywords': [('phone', 15), ('app', 12), ('tech', 10)]
        }
    }
    
    visualizer = Visualizer()
    
    # Test plots
    fig1 = visualizer.plot_sentiment_distribution(sample_sentiment_data)
    fig2 = visualizer.plot_topic_comparison(sample_sentiment_data)
    fig3 = visualizer.create_topic_wordclouds(sample_topic_trends)
    
    plt.show()
