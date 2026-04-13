import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Union

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Also try the older punkt version as fallback
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        pass

class TextPreprocessor:
    """Handles text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, hashtags, and special characters.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ mentions (but keep hashtags for topic identification)
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters but keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        # Keep some important stopwords that might be meaningful for sentiment
        important_stopwords = {'not', 'no', 'nor', 'never', 'none'}
        return [token for token in tokens if token not in self.stop_words or token in important_stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                           new_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to preprocess
            new_column: Name of the new column to store preprocessed text
            
        Returns:
            DataFrame with preprocessed text
        """
        df_copy = df.copy()
        
        print(f"Preprocessing {len(df_copy)} texts...")
        
        # Apply preprocessing to each text
        df_copy[new_column] = df_copy[text_column].apply(self.preprocess_text)
        
        # Remove rows where cleaned text is empty OR too short (less than 2 characters)
        df_copy = df_copy[df_copy[new_column].str.len() > 2].reset_index(drop=True)
        
        print(f"Preprocessing complete! {len(df_copy)} texts remaining after cleaning.")
        
        # Add debugging info
        if len(df_copy) < len(df_copy) * 0.5:  # If we lost more than 50% of data
            print("⚠️  Warning: Lost more than 50% of data during preprocessing")
            print("Sample of cleaned texts:")
            for i in range(min(3, len(df_copy))):
                print(f"  {i+1}. '{df_copy.iloc[i][new_column]}'")
        
        return df_copy
    
    def get_preprocessing_stats(self, original_df: pd.DataFrame, 
                              processed_df: pd.DataFrame) -> dict:
        """
        Get statistics about the preprocessing process.
        
        Args:
            original_df: Original DataFrame before preprocessing
            processed_df: Processed DataFrame after preprocessing
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'original_count': len(original_df),
            'processed_count': len(processed_df),
            'removed_count': len(original_df) - len(processed_df),
            'removal_percentage': ((len(original_df) - len(processed_df)) / len(original_df)) * 100
        }
        
        if 'cleaned_text' in processed_df.columns:
            # Calculate average text length before and after
            original_avg_length = original_df['text'].astype(str).str.len().mean()
            processed_avg_length = processed_df['cleaned_text'].str.len().mean()
            
            stats.update({
                'original_avg_length': original_avg_length,
                'processed_avg_length': processed_avg_length,
                'length_reduction': ((original_avg_length - processed_avg_length) / original_avg_length) * 100
            })
        
        return stats

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with sample texts
    sample_texts = [
        "I love this movie! It's amazing #movie",
        "This phone is terrible, don't buy it @company",
        "Just finished reading a great book! Highly recommend it.",
        "The cricket match was exciting today! #sports"
    ]
    
    print("Testing text preprocessing:")
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.preprocess_text(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned: {cleaned}")
