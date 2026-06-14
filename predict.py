"""
Predict sentiment for Arabic text
"""

import numpy as np
import pickle
import joblib
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from utils import preprocess_text

class ArabicSentimentAnalyzer:
    def __init__(self, model_path='models/'):
        """Load all model artifacts"""
        print("Loading sentiment analyzer...")
        
        # Load neural network model
        self.model = load_model(f'{model_path}sentiment_model.h5')
        
        # Load Word2Vec model
        self.w2v_model = Word2Vec.load(f'{model_path}word2vec.model')
        
        # Load TF-IDF vectorizer and IDF dict
        self.tfidf_vectorizer = joblib.load(f'{model_path}tfidf_vectorizer.pkl')
        
        with open(f'{model_path}idf_dict.pkl', 'rb') as f:
            self.idf_dict = pickle.load(f)
        
        self.sentiment_map = {0: 'Negative 😡', 1: 'Neutral 😐', 2: 'Positive 😊'}
        
        print("✅ Model loaded successfully!")
    
    def get_weighted_doc_vector(self, text):
        """Get weighted document vector"""
        words = text.split()
        vecs = []
        weights = []
        
        for w in words:
            if w in self.w2v_model.wv and w in self.idf_dict:
                vecs.append(self.w2v_model.wv[w] * self.idf_dict[w])
                weights.append(self.idf_dict[w])
        
        if len(vecs) == 0:
            return np.zeros(self.w2v_model.vector_size)
        
        return np.sum(vecs, axis=0) / np.sum(weights)
    
    def predict(self, text):
        """Predict sentiment of Arabic text"""
        # Preprocess text
        clean_text = preprocess_text(text)
        
        if not clean_text:
            return "Could not classify (text too short or filtered)"
        
        # Create embedding
        embedding = self.get_weighted_doc_vector(clean_text)
        embedding = embedding.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(embedding, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        return {
            'sentiment': self.sentiment_map[predicted_class],
            'confidence': float(confidence),
            'class': predicted_class
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# CLI interface
if __name__ == "__main__":
    analyzer = ArabicSentimentAnalyzer()
    
    print("\n" + "="*50)
    print("Arabic Sentiment Analysis CLI")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        text = input("Enter Arabic text: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if text:
            result = analyzer.predict(text)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2%}\n")
