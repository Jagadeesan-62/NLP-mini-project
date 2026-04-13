import requests
import json

def test_flask_api():
    """Test Flask API endpoints."""
    base_url = "http://localhost:5000"
    
    print("Testing Flask API endpoints...")
    print("=" * 50)
    
    # Test 1: Home page
    try:
        response = requests.get(f"{base_url}/")
        print(f"1. Home Page: {response.status_code} - {'OK' if response.status_code == 200 else 'FAILED'}")
    except Exception as e:
        print(f"1. Home Page: FAILED - {e}")
    
    # Test 2: Dataset page
    try:
        response = requests.get(f"{base_url}/dataset")
        print(f"2. Dataset Page: {response.status_code} - {'OK' if response.status_code == 200 else 'FAILED'}")
    except Exception as e:
        print(f"2. Dataset Page: FAILED - {e}")
    
    # Test 3: Search page
    try:
        response = requests.get(f"{base_url}/search")
        print(f"3. Search Page: {response.status_code} - {'OK' if response.status_code == 200 else 'FAILED'}")
    except Exception as e:
        print(f"3. Search Page: FAILED - {e}")
    
    # Test 4: Analyze keyword API
    try:
        response = requests.post(f"{base_url}/analyze", 
                               data={"keyword": "iphone"},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code == 200:
            data = response.json()
            print(f"4. Analyze API: {response.status_code} - OK")
            print(f"   Keyword: {data.get('keyword', 'N/A')}")
            print(f"   Total Tweets: {data.get('sentiment_data', {}).get('total_count', 'N/A')}")
            print(f"   Positive %: {data.get('sentiment_data', {}).get('positive_percentage', 'N/A')}%")
        else:
            print(f"4. Analyze API: {response.status_code} - FAILED")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"4. Analyze API: FAILED - {e}")
    
    # Test 5: Comparison API
    try:
        response = requests.post(f"{base_url}/compare",
                               data={"keywords": ["iphone", "samsung"]},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code == 200:
            data = response.json()
            print(f"5. Comparison API: {response.status_code} - OK")
            print(f"   Keywords compared: {list(data.get('keywords_data', {}).keys())}")
        else:
            print(f"5. Comparison API: {response.status_code} - FAILED")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"5. Comparison API: FAILED - {e}")
    
    # Test 6: Insights API
    try:
        response = requests.post(f"{base_url}/generate_insights",
                               data={"keywords": ["iphone", "samsung", "cricket"]},
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code == 200:
            data = response.json()
            print(f"6. Insights API: {response.status_code} - OK")
            print(f"   Most Positive: {data.get('most_positive', ['N/A'])[0]}")
            print(f"   Most Negative: {data.get('most_negative', ['N/A'])[0]}")
        else:
            print(f"6. Insights API: {response.status_code} - FAILED")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"6. Insights API: FAILED - {e}")
    
    print("=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    test_flask_api()
