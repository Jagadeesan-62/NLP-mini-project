#!/usr/bin/env python3
"""
Troubleshooting script for common issues with the sentiment analysis application
"""

import os
import pandas as pd
from data_handler import DataHandler
from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer

def troubleshoot_data_loading():
    """Troubleshoot data loading issues."""
    print("🔍 Data Loading Troubleshooting")
    print("=" * 50)
    
    # Check dataset paths
    paths_to_check = [
        "data/sentiment140.csv",
        "sentiment140.csv",
        "./data/sentiment140.csv",
        "./sentiment140.csv"
    ]
    
    found_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ Dataset found: {path}")
            found_path = path
            break
        else:
            print(f"❌ Not found: {path}")
    
    if not found_path:
        print("\n❌ No dataset found!")
        print("Please download the Sentiment140 dataset:")
        print("1. Go to: http://help.sentiment140.com/for-students")
        print("2. Download the training.1600000.processed.noemoticon.csv file")
        print("3. Rename it to 'sentiment140.csv'")
        print("4. Place it in the project root or 'data/' folder")
        return False
    
    # Try to load a small sample
    try:
        handler = DataHandler(found_path)
        df = handler.load_data(sample_size=100)
        print(f"✅ Successfully loaded {len(df)} rows")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Sentiment values: {df['sentiment'].unique()}")
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

def troubleshoot_preprocessing():
    """Troubleshoot preprocessing issues."""
    print("\n🧹 Preprocessing Troubleshooting")
    print("=" * 50)
    
    try:
        # Load sample data
        handler = DataHandler()
        df = handler.load_data(sample_size=1000)
        
        # Test preprocessing
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df, 'text', 'cleaned_text')
        
        print(f"✅ Original rows: {len(df)}")
        print(f"✅ Processed rows: {len(df_processed)}")
        print(f"✅ Average text length: {df_processed['cleaned_text'].str.len().mean():.1f}")
        
        # Show some examples
        print("\n📝 Preprocessing Examples:")
        for i in range(min(3, len(df_processed))):
            original = df.iloc[i]['text'][:50]
            cleaned = df_processed.iloc[i]['cleaned_text']
            print(f"Original: {original}...")
            print(f"Cleaned:  {cleaned}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        return False

def troubleshoot_feature_engineering():
    """Troubleshoot feature engineering issues."""
    print("\n⚙️  Feature Engineering Troubleshooting")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        handler = DataHandler()
        df = handler.load_data(sample_size=2000)
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df, 'text', 'cleaned_text')
        
        # Test TF-IDF
        fe = FeatureEngineer(max_features=1000)  # Smaller for testing
        X = fe.fit_transform(df_processed['cleaned_text'])
        
        print(f"✅ TF-IDF matrix shape: {X.shape}")
        print(f"✅ Number of features: {len(fe.feature_names)}")
        print(f"✅ Sample features: {list(fe.feature_names[:10])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {str(e)}")
        
        if "empty vocabulary" in str(e).lower():
            print("\n💡 Empty Vocabulary Solutions:")
            print("1. Increase sample size (try 100,000 instead of 10,000)")
            print("2. Check if preprocessing is too aggressive")
            print("3. Verify dataset contains English text")
            print("4. Try reducing min_df parameter in TF-IDF")
        
        return False

def main():
    """Run all troubleshooting checks."""
    print("🚀 Sentiment Analysis Application Troubleshooter")
    print("=" * 60)
    
    # Run all checks
    data_ok = troubleshoot_data_loading()
    if data_ok:
        preprocess_ok = troubleshoot_preprocessing()
        if preprocess_ok:
            feature_ok = troubleshoot_feature_engineering()
            
            print("\n" + "=" * 60)
            if data_ok and preprocess_ok and feature_ok:
                print("🎉 All checks passed! The application should work correctly.")
                print("\nTo run the application:")
                print("streamlit run app.py")
            else:
                print("❌ Some issues found. Please follow the suggestions above.")
        else:
            print("❌ Preprocessing issues found.")
    else:
        print("❌ Data loading issues found.")
    
    print("\n" + "=" * 60)
    print("If you still have issues:")
    print("1. Check that all requirements are installed: pip install -r requirements.txt")
    print("2. Ensure you have enough RAM (at least 4GB recommended)")
    print("3. Try running with a smaller sample size")
    print("4. Check the dataset file is not corrupted")

if __name__ == "__main__":
    main()
