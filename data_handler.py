import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io
from typing import Tuple, Optional

class DataHandler:
    """Handles loading and preprocessing of the Sentiment140 dataset from Google Drive."""
    
    def __init__(self, data_path: str = None):
        # Google Drive link for the dataset
        self.drive_url = "https://drive.google.com/file/d/1vmKPAU-nmBxl9Bo9pHqBl8yS74rx_rny/view?usp=sharing"
        self.direct_download_url = "https://drive.google.com/uc?export=download&id=1vmKPAU-nmBxl9Bo9pHqBl8yS74rx_rny"
        
        # Local cache path
        self.cache_path = "data/sentiment140_cache.csv"
        self.data_path = self.cache_path
        
        self.data = None
        # Will determine columns based on actual file format
        self.columns = None
    
    def download_from_drive(self) -> str:
        """
        Download dataset from Google Drive.
        
        Returns:
            Path to downloaded file
        """
        print("Downloading dataset from Google Drive...")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        try:
            # Download the file
            response = requests.get(self.direct_download_url, stream=True)
            response.raise_for_status()
            
            # Save to cache
            with open(self.cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Dataset downloaded successfully to: {self.cache_path}")
            return self.cache_path
            
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            raise
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Sentiment140 dataset from Google Drive or cache.
        
        Args:
            sample_size: If provided, load only this many rows for faster processing
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Check if cached file exists, if not download from Drive
            if not os.path.exists(self.cache_path):
                print("Dataset not found in cache, downloading from Google Drive...")
                self.download_from_drive()
            else:
                print(f"Using cached dataset: {self.cache_path}")
            
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
    except Exception as e:
        print(f"Error: {e}")
        print("The dataset will be automatically downloaded from Google Drive when you run the application.")
        print("If download fails, you can manually download from:")
        print(f"https://drive.google.com/file/d/1vmKPAU-nmBxl9Bo9pHqBl8yS74rx_rny/view?usp=sharing")
