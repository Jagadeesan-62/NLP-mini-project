import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

class DataHandler:
    """Handles loading and preprocessing of the Sentiment140 dataset."""
    
    def __init__(self, data_path: str = None):
        # Try multiple possible paths for the dataset
        if data_path is None:
            possible_paths = [
                "data/sentiment140.csv",
                "sentiment140.csv",
                "./data/sentiment140.csv",
                "./sentiment140.csv"
            ]
            self.data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    break
        else:
            self.data_path = data_path
            
        self.data = None
        # Will determine columns based on actual file format
        self.columns = None
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Sentiment140 dataset.
        
        Args:
            sample_size: If provided, load only this many rows for faster processing
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Check if we found a valid path
            if self.data_path is None:
                raise FileNotFoundError("Dataset not found. Please place sentiment140.csv in the project root or data/ folder")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
            print(f"Loading dataset from: {self.data_path}")
            
            # First, check the actual format of the CSV
            with open(self.data_path, 'r', encoding='latin-1') as f:
                first_line = f.readline().strip()
                print(f"First line of CSV: {first_line}")
            
            # Determine the format and load accordingly
            if 'sentence' in first_line.lower():
                # Format: sentence,sentiment
                print("Detected 2-column format (sentence,sentiment)")
                df = pd.read_csv(self.data_path, 
                               encoding='latin-1', 
                               header=None, 
                               names=['text', 'sentiment'])
                
                # Convert sentiment: handle string values
                # In this format, sentiment might be string values
                def convert_sentiment(x):
                    try:
                        x_int = int(x)
                        return 1 if x_int > 0 else 0
                    except (ValueError, TypeError):
                        return 0  # Default to negative if conversion fails
                
                df['sentiment'] = df['sentiment'].apply(convert_sentiment)
                
            else:
                # Standard 6-column format
                print("Detected 6-column format")
                self.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
                df = pd.read_csv(self.data_path, 
                               encoding='latin-1', 
                               header=None, 
                               names=self.columns)
                
                # Convert sentiment labels: 0 -> 0 (Negative), 4 -> 1 (Positive)
                df['sentiment'] = df['sentiment'].replace(4, 1)
            
            # Handle NaN values in text column
            if 'text' in df.columns:
                df['text'] = df['text'].fillna('')
            
            # Sample data if requested (for faster processing during development)
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Convert date to datetime if column exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            self.data = df
            print(f"Data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
            if 'text' in df.columns:
                print(f"Text column NaN count: {df['text'].isna().sum()}")
                print(f"Sample text: {df.iloc[0]['text'][:100]}...")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """Get sample rows from the dataset."""
        if self.data is None:
            self.load_data()
        return self.data.head(n)
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset."""
        if self.data is None:
            self.load_data()
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'sentiment_distribution': self.data['sentiment'].value_counts().to_dict(),
            'null_values': self.data.isnull().sum().to_dict(),
            'date_range': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max()
            } if 'date' in self.data.columns else None
        }
        return info
    
    def create_sample_dataset(self, output_path: str = "data/sentiment140_sample.csv", 
                            sample_size: int = 10000) -> None:
        """Create a smaller sample dataset for faster processing."""
        if self.data is None:
            self.load_data()
        
        sample_df = self.data.sample(n=sample_size, random_state=42)
        sample_df.to_csv(output_path, index=False)
        print(f"Sample dataset created at {output_path} with {sample_size} rows")

if __name__ == "__main__":
    # Test the data handler
    handler = DataHandler()
    
    # Try to load a small sample first
    try:
        df = handler.load_data(sample_size=1000)
        print("\nSample data:")
        print(df.head())
        print("\nData info:")
        print(handler.get_data_info())
    except FileNotFoundError:
        print("Please download the Sentiment140 dataset and place it in the data/ folder")
        print("Download from: http://help.sentiment140.com/for-students")
