# Topic-wise Social Media Trend and Sentiment Analysis

A comprehensive NLP-based project that analyzes Twitter sentiment across different topics using Machine Learning with an interactive Streamlit dashboard.

## Features

- **Sentiment Analysis**: Classify tweets as Positive or Negative using Multinomial Naive Bayes
- **Topic Identification**: Automatically categorize tweets into Movies, Sports, or Technology
- **Trend Analysis**: Identify trending keywords and topics
- **Interactive Visualizations**: Charts, word clouds, and comprehensive analytics
- **Multi-page Dashboard**: Clean, responsive Streamlit interface

## Project Structure

```
project/
|-- app.py                    # Main Streamlit application
|-- data_handler.py           # Data loading and handling
|-- preprocessing.py          # Text cleaning and preprocessing
|-- feature_engineering.py     # TF-IDF vectorization
|-- model.py                  # ML model training and prediction
|-- topic_analyzer.py         # Topic identification and analysis
|-- visualization.py          # Charts and visualizations
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation
|-- data/
|   |-- sentiment140.csv       # Dataset (download required)
|-- pages/
|   |-- home.py               # Home page
|   |-- dataset.py            # Dataset overview
|   |-- topic_analysis.py     # Topic analysis page
|   |-- comparison.py         # Topic comparison page
|   |-- insights.py           # Insights and recommendations
```

## Dataset

This project uses the **Sentiment140** dataset, which contains 1.6 million labeled tweets.

### Automatic Download
The dataset is automatically downloaded from Google Drive when you first run the application:
- **Download Link**: https://drive.google.com/file/d/1vmKPAU-nmBxl9Bo9pHqBl8yS74rx_rny/view?usp=sharing
- **Cache Location**: `data/sentiment140_cache.csv`
- **Size**: ~80MB (compressed)

### Manual Download (Optional)
If automatic download fails, you can manually download:
1. Visit: https://drive.google.com/file/d/1vmKPAU-nmBxl9Bo9pHqBl8yS74rx_rny/view?usp=sharing
2. Download the file and save as `data/sentiment140_cache.csv`
3. The dataset should have the following columns (in order):
   - sentiment (0=negative, 4=positive)
   - id
   - date
   - query
   - user
   - text

## Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset and place it in the `data/` folder as described above

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Application Pages

### 1. Home
- Project overview and introduction
- Model performance metrics
- Quick insights
- Real-time sentiment prediction demo

### 2. Dataset Overview
- Dataset statistics and information
- Sentiment distribution
- Sample data preview
- Preprocessing statistics
- Data quality metrics

### 3. Topic Analysis
- Select and analyze specific topics (Movies, Sports, Technology)
- Topic-wise sentiment breakdown
- Top keywords and word clouds
- Sample tweets for each topic
- Temporal sentiment trends

### 4. Comparison
- Side-by-side topic comparison
- Sentiment distribution across topics
- Statistical analysis and correlations
- Engagement metrics
- Performance radar charts

### 5. Insights
- Automatically generated insights
- Trending topics ranking
- Performance recommendations
- Action items for strategic planning

## Technical Implementation

### Data Processing
- **Text Cleaning**: Removes URLs, mentions, hashtags, special characters
- **Tokenization**: Splits text into individual words
- **Stopword Removal**: Eliminates common English stopwords
- **Lemmatization**: Reduces words to their base form

### Feature Engineering
- **TF-IDF Vectorization**: Converts text to numerical features
- **Feature Limiting**: Uses top 5,000 most important features
- **N-gram Support**: Includes both unigrams and bigrams

### Machine Learning Model
- **Algorithm**: Multinomial Naive Bayes
- **Training**: 80% training, 20% testing split
- **Evaluation**: Accuracy score, confusion matrix, cross-validation
- **Prediction**: Real-time sentiment classification

### Topic Identification
- **Keyword-based**: Uses predefined keyword sets for each topic
- **Topics Supported**:
  - Movies: movie, film, actor, actress, director, cinema, theater, hollywood
  - Sports: cricket, match, team, game, player, sport, football, basketball, tennis, IPL
  - Technology: phone, mobile, AI, app, software, computer, laptop, tech, digital

### Visualization
- **Charts**: Bar charts, pie charts, line plots, radar charts
- **Word Clouds**: Topic-specific word frequency visualization
- **Statistical Plots**: Correlation matrices, trend analysis
- **Interactive Elements**: Streamlit metrics, data tables, and filters

## Performance

The system typically achieves:
- **Accuracy**: ~75-80% on sentiment classification
- **Processing Speed**: Analyzes 10,000 tweets in <30 seconds
- **Memory Usage**: Optimized for efficient processing
- **Scalability**: Can handle larger datasets with appropriate hardware

## Customization

### Adding New Topics
1. Update the `topic_keywords` dictionary in `topic_analyzer.py`
2. Add new topic keywords and descriptions
3. The system will automatically include them in analysis

### Modifying the Model
1. Experiment with different algorithms in `model.py`
2. Adjust hyperparameters for better performance
3. Add ensemble methods for improved accuracy

### Enhancing Preprocessing
1. Add new cleaning rules in `preprocessing.py`
2. Implement advanced NLP techniques
3. Add language detection for multilingual support

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure `sentiment140.csv` is in the `data/` folder
   - Check file permissions and path

2. **Memory Issues**
   - Reduce sample size in `data_handler.py`
   - Close other applications to free memory

3. **Slow Performance**
   - Use smaller sample sizes for testing
   - Ensure all dependencies are up to date

4. **Visualization Errors**
   - Check matplotlib backend settings
   - Verify all required packages are installed

### Dependencies Issues
If you encounter dependency conflicts, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Future Enhancements

- **Real-time Twitter API Integration**
- **Advanced NLP Models (BERT, RoBERTa)**
- **Multi-language Support**
- **Sentiment Intensity Scoring**
- **User Engagement Analytics**
- **Historical Trend Analysis**
- **Export Functionality (PDF, Excel)**
- **Alert System for Sentiment Changes**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with Twitter's terms of service and data usage policies when using real Twitter data.

## Contact

For questions or suggestions, please create an issue in the repository or contact the project maintainer.

---

**Note**: This project is designed for educational and demonstration purposes. The sentiment analysis model is trained on the Sentiment140 dataset and may not perform optimally on all types of text data. For production use, consider training on domain-specific data and implementing additional validation mechanisms.
