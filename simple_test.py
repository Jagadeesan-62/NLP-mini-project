#!/usr/bin/env python3
"""
Simple test to debug preprocessing issues
"""

import pandas as pd
import re
from data_handler import DataHandler

def simple_clean_text(text):
    """Very simple text cleaning for debugging."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the text)
    text = re.sub(r'#', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def test_simple_preprocessing():
    """Test with simple preprocessing."""
    print("🧪 Testing Simple Preprocessing")
    print("=" * 50)
    
    # Load data
    handler = DataHandler()
    df = handler.load_data(sample_size=100)
    
    print(f"Loaded {len(df)} rows")
    
    # Apply simple cleaning
    df['simple_clean'] = df['text'].apply(simple_clean_text)
    
    # Show examples
    print("\n📝 Sample preprocessing results:")
    for i in range(min(5, len(df))):
        original = str(df.iloc[i]['text'])[:60]
        cleaned = str(df.iloc[i]['simple_clean'])
        print(f"Original: {original}...")
        print(f"Cleaned:  {cleaned}")
        print()
    
    # Check lengths
    print(f"Average original length: {df['text'].str.len().mean():.1f}")
    print(f"Average cleaned length: {df['simple_clean'].str.len().mean():.1f}")
    
    # Filter out empty/short texts
    df_filtered = df[df['simple_clean'].str.len() > 5]
    print(f"Rows after filtering (length > 5): {len(df_filtered)}")
    
    if len(df_filtered) > 0:
        print("✅ Simple preprocessing works!")
        return True
    else:
        print("❌ Simple preprocessing failed!")
        return False

if __name__ == "__main__":
    test_simple_preprocessing()
