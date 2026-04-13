#!/usr/bin/env python3
"""
Check the CSV file structure directly
"""

import pandas as pd

def check_csv_structure():
    """Check CSV file structure without column names."""
    print("🔍 Checking CSV File Structure")
    print("=" * 50)
    
    # Try reading without column names first
    try:
        # Read first few lines without headers
        with open("data/sentiment140.csv", 'r', encoding='latin-1') as f:
            lines = []
            for i, line in enumerate(f):
                if i < 5:
                    lines.append(line.strip())
                    print(f"Line {i+1}: {line[:100]}...")
                else:
                    break
        
        print(f"\nNumber of columns in first line: {len(lines[0].split(','))}")
        
        # Now try reading with proper column assignment
        columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv("data/sentiment140.csv", 
                       encoding='latin-1', 
                       header=None, 
                       names=columns,
                       nrows=5)
        
        print(f"\nColumns after reading: {list(df.columns)}")
        print("Sample data:")
        print(df)
        
        # Check text column specifically
        print(f"\nText column values:")
        for i, text in enumerate(df['text']):
            print(f"Row {i}: '{text}' (length: {len(str(text))})")
            
    except Exception as e:
        print(f"Error: {e}")
        
        # Try alternative approach
        print("\nTrying alternative reading approach...")
        try:
            df = pd.read_csv("data/sentiment140.csv", encoding='latin-1', header=None)
            print(f"Shape without column names: {df.shape}")
            print("First few rows:")
            print(df.head())
        except Exception as e2:
            print(f"Alternative error: {e2}")

if __name__ == "__main__":
    check_csv_structure()
