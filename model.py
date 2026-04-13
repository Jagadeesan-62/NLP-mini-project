import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import pickle

class SentimentModel:
    """Handles sentiment analysis model training and prediction."""
    
    def __init__(self, model_path: str = "models/sentiment_model.pkl"):
        self.model = None
        self.model_path = model_path
        self.is_trained = False
    
    def create_model(self, alpha: float = 1.0) -> MultinomialNB:
        """
        Create a Multinomial Naive Bayes model.
        
        Args:
            alpha: Laplace smoothing parameter
            
        Returns:
            Configured MultinomialNB model
        """
        model = MultinomialNB(alpha=alpha)
        return model
    
    def train(self, X_train: np.ndarray, y_train: pd.Series, 
              alpha: float = 1.0) -> None:
        """
        Train the sentiment analysis model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            alpha: Laplace smoothing parameter
        """
        print("Training sentiment analysis model...")
        
        self.model = self.create_model(alpha=alpha)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Model training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
        
        evaluation_results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return evaluation_results
    
    def predict_sentiment(self, text: str, vectorizer) -> Tuple[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            vectorizer: Fitted TF-IDF vectorizer
            
        Returns:
            Tuple of (predicted_sentiment, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Transform text using vectorizer
        text_vector = vectorizer.transform([text])
        
        # Make prediction
        prediction = self.predict(text_vector)[0]
        probabilities = self.predict_proba(text_vector)[0]
        
        # Convert to sentiment label and confidence
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probabilities)
        
        return sentiment, confidence
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (uses model_path if None)
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        save_path = filepath or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from (uses model_path if None)
        """
        load_path = filepath or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self.model = joblib.load(load_path)
        self.is_trained = True
        print(f"Model loaded from {load_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            save_path: str = "models/confusion_matrix.png") -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def get_feature_importance(self, feature_names: np.ndarray, 
                             top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each class.
        
        Args:
            feature_names: Array of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get log probabilities for each class
        log_prob = self.model.feature_log_prob_
        
        feature_importance = {}
        
        for i, class_name in enumerate(['Negative', 'Positive']):
            # Get top features for this class
            top_indices = np.argsort(log_prob[i])[::-1][:top_n]
            top_features = [
                (feature_names[idx], log_prob[i][idx]) 
                for idx in top_indices
            ]
            feature_importance[class_name] = top_features
        
        return feature_importance

if __name__ == "__main__":
    # Test the model
    from preprocessing import TextPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Sample data for testing
    sample_texts = [
        "I love this movie it's amazing",
        "This phone is terrible don't buy it",
        "Great cricket match today exciting",
        "The film was boring and disappointing",
        "Amazing product! Highly recommend",
        "Worst experience ever, very disappointed"
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned_texts = [preprocessor.preprocess_text(text) for text in sample_texts]
    
    # Feature engineering
    fe = FeatureEngineer(max_features=100)
    X = fe.fit_transform(pd.Series(cleaned_texts))
    y = pd.Series(sample_labels)
    
    # Train model
    model = SentimentModel()
    model.train(X, y)
    
    # Evaluate
    X_train, X_test, y_train, y_test = fe.prepare_train_test_split(X, y, test_size=0.3)
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    print("\nTest prediction:")
    test_text = "This is an amazing product!"
    sentiment, confidence = model.predict_sentiment(test_text, fe.vectorizer)
    print(f"Text: {test_text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
