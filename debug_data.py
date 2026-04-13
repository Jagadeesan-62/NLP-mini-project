#!/usr/bin/env python3
"""
Debug the actual data structure
"""

import pandas as pd
from data_handler import DataHandler

def debug_data_structure():
    """Debug the actual data structure."""
    print("🔍 Debugging Data Structure")
    print("=" * 50)
    
    # Load data
    handler = DataHandler()
    df = handler.load_data(sample_size=10)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Column types:\n{df.dtypes}")
    print()
    
    # Show first few rows
    print("First 5 rows:")
    print(df.head())
    print()
    
    # Check text column specifically
    print("Text column analysis:")
    print(f"Text column type: {type(df['text'])}")
    print(f"First text value: {repr(df.iloc[0]['text'])}")
    print(f"Text is null: {df['text'].isnull().any()}")
    print(f"Text length: {len(str(df.iloc[0]['text']))}")
    
    # Check if there are issues with column names
    print(f"\nActual column names from file: {handler.columns}")

if __name__ == "__main__":
    debug_data_structure()
