#!/usr/bin/env python3
"""
Test script to verify dataset loading and basic functionality
"""

import os
import pandas as pd
from data_handler import DataHandler

def test_dataset_loading():
    """Test if the dataset can be loaded properly."""
    print("🔍 Testing dataset loading...")
    
    # Check if dataset exists
    possible_paths = [
        "data/sentiment140.csv",
        "sentiment140.csv",
        "./data/sentiment140.csv",
        "./sentiment140.csv"
    ]
    
    dataset_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found at: {path}")
            dataset_found = True
            break
    
    if not dataset_found:
        print("❌ Dataset not found!")
        print("Please download the Sentiment140 dataset and place it as 'sentiment140.csv' in:")
        print("- Project root directory")
        print("- Or in the 'data/' folder")
        print("\nDownload from: http://help.sentiment140.com/for-students")
        return False
    
    # Test data loading
    try:
        handler = DataHandler()
        print(f"✅ DataHandler initialized with path: {handler.data_path}")
        
        # Load a small sample
        df = handler.load_data(sample_size=100)
        print(f"✅ Successfully loaded {len(df)} rows")
        
        # Check columns
        expected_columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        if all(col in df.columns for col in expected_columns):
            print("✅ All expected columns present")
        else:
            print("❌ Missing columns")
            return False
        
        # Check sentiment values
        unique_sentiments = df['sentiment'].unique()
        print(f"✅ Sentiment values: {unique_sentiments}")
        
        # Show sample data
        print("\n📋 Sample data:")
        print(df[['sentiment', 'text']].head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n🔍 Testing dependencies...")
    
    dependencies = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('streamlit', 'st'),
        ('sklearn', None),
        ('matplotlib', 'plt'),
        ('seaborn', None),
        ('wordcloud', None),
        ('nltk', None)
    ]
    
    all_good = True
    for dep, alias in dependencies:
        try:
            if alias:
                exec(f"import {dep} as {alias}")
            else:
                exec(f"import {dep}")
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - Please install: pip install {dep}")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("🚀 Running setup tests...\n")
    
    deps_ok = test_dependencies()
    data_ok = test_dataset_loading()
    
    print("\n" + "="*50)
    if deps_ok and data_ok:
        print("🎉 All tests passed! You can run the application with:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    print("="*50)

if __name__ == "__main__":
    main()
