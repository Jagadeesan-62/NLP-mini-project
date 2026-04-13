import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import joblib
import os

class FeatureEngineer:
    """Handles feature engineering for text data using TF-IDF."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None
    
    def create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Create and configure TF-IDF vectorizer.
        
        Returns:
            Configured TfidfVectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,  # Reduced from 2 to 1 to avoid empty vocabulary
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            token_pattern=r'(?u)\b\w+\b'  # Better token pattern
        )
        
        return vectorizer
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            TF-IDF matrix
        """
        # Check if texts are empty or too short
        if texts.empty or texts.str.len().sum() == 0:
            raise ValueError("No valid text data for TF-IDF vectorization")
        
        print(f"Processing {len(texts)} documents for TF-IDF...")
        print(f"Average text length: {texts.str.len().mean():.1f} characters")
        
        self.vectorizer = self.create_tfidf_vectorizer()
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            print(f"Number of features: {len(self.feature_names)}")
            
            if len(self.feature_names) == 0:
                raise ValueError("Empty vocabulary after TF-IDF vectorization")
            
            return tfidf_matrix.toarray()
            
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print("Empty vocabulary detected. Trying with more lenient settings...")
                # Try with more lenient settings
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=(1, 1),  # Only unigrams
                    lowercase=True,
                    min_df=1,
                    max_df=1.0,
                    token_pattern=r'(?u)\b\w+\b'
                )
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
                print(f"Retried TF-IDF matrix shape: {tfidf_matrix.shape}")
                print(f"Retried number of features: {len(self.feature_names)}")
                return tfidf_matrix.toarray()
            else:
                raise e
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            TF-IDF matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        return tfidf_matrix.toarray()
    
    def get_top_features(self, tfidf_matrix: np.ndarray, 
                        feature_names: np.ndarray, 
                        top_n: int = 10) -> Dict[str, float]:
        """
        Get top features based on average TF-IDF score.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: Array of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features and their scores
        """
        # Calculate average TF-IDF score for each feature
        mean_scores = np.mean(tfidf_matrix, axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_scores)[::-1][:top_n]
        top_features = {
            feature_names[i]: mean_scores[i] 
            for i in top_indices
        }
        
        return top_features
    
    def prepare_train_test_split(self, X: np.ndarray, y: pd.Series, 
                               test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training set sentiment distribution:\n{y_train.value_counts()}")
        print(f"Test set sentiment distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_vectorizer(self, filepath: str = "models/tfidf_vectorizer.pkl") -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save. Fit the vectorizer first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str = "models/tfidf_vectorizer.pkl") -> None:
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to load the vectorizer from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Vectorizer loaded from {filepath}")
    
    def get_feature_importance(self, tfidf_matrix: np.ndarray, 
                             labels: pd.Series, 
                             top_n: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance for each class.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            labels: Target labels
            top_n: Number of top features per class
            
        Returns:
            Dictionary with top features for each class
        """
        feature_importance = {}
        
        for label in np.unique(labels):
            # Get indices for current class
            class_indices = labels[labels == label].index
            class_matrix = tfidf_matrix[class_indices]
            
            # Calculate average TF-IDF for this class
            class_mean_scores = np.mean(class_matrix, axis=0)
            
            # Get top features for this class
            top_indices = np.argsort(class_mean_scores)[::-1][:top_n]
            top_features = {
                self.feature_names[i]: class_mean_scores[i] 
                for i in top_indices
            }
            
            feature_importance[f'class_{label}'] = top_features
        
        return feature_importance

if __name__ == "__main__":
    # Test the feature engineer
    from preprocessing import TextPreprocessor
    
    # Sample data
    sample_texts = [
        "I love this movie it's amazing",
        "This phone is terrible don't buy it",
        "Great cricket match today exciting",
        "The film was boring and disappointing"
    ]
    
    # Preprocess texts
    preprocessor = TextPreprocessor()
    cleaned_texts = [preprocessor.preprocess_text(text) for text in sample_texts]
    
    # Create TF-IDF features
    fe = FeatureEngineer(max_features=100)
    tfidf_matrix = fe.fit_transform(pd.Series(cleaned_texts))
    
    print("\nTop features:")
    top_features = fe.get_top_features(tfidf_matrix, fe.feature_names, top_n=5)
    for feature, score in top_features.items():
        print(f"{feature}: {score:.4f}")
