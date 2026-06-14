"""
Train Arabic Sentiment Analysis Model
Hybrid Word2Vec + TF-IDF embeddings with Neural Network
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_text

# ============ CONFIGURATION ============
DATA_PATH = "/content/CompanyReviews.csv"  # Update this path
TEST_SIZE = 0.2
VAL_SIZE = 0.176  # 0.176 of train = 0.14 of total
RANDOM_STATE = 42
WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def load_data(filepath):
    """Load and prepare data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop unnamed column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Drop missing values
    df.dropna(inplace=True)
    
    return df

def preprocess_corpus(df, text_column='review_description'):
    """Preprocess all texts"""
    print("Preprocessing texts...")
    df['clean_text'] = df[text_column].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['clean_text'].str.len() > 0]
    print(f"After preprocessing: {len(df)} samples")
    
    return df

def train_word2vec(tokenized_docs):
    """Train Word2Vec model"""
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=4,
        sg=1  # Skip-gram
    )
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    return w2v_model

def get_weighted_doc_vector(doc, w2v_model, idf_dict):
    """Get weighted document vector using TF-IDF + Word2Vec"""
    words = doc.split()
    vecs = []
    weights = []
    
    for w in words:
        if w in w2v_model.wv and w in idf_dict:
            vecs.append(w2v_model.wv[w] * idf_dict[w])
            weights.append(idf_dict[w])
    
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    
    return np.sum(vecs, axis=0) / np.sum(weights)

def create_hybrid_embeddings(df, w2v_model, tfidf_vectorizer, idf_dict):
    """Create hybrid TF-IDF + Word2Vec embeddings"""
    print("Creating hybrid embeddings...")
    tokenized_docs = [text.split() for text in df['clean_text']]
    
    X = np.array([get_weighted_doc_vector(text, w2v_model, idf_dict) 
                  for text in df['clean_text']])
    
    return X

def build_model(input_dim, num_classes=3):
    """Build neural network classifier"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    """Main training function"""
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Preprocess texts
    df = preprocess_corpus(df)
    
    # Prepare labels (map ratings: -1=Negative, 0=Neutral, 1=Positive)
    # Assuming rating column values: -1, 0, 1
    y = df['rating'].values
    
    # Remap labels to 0, 1, 2
    y_remapped = y.copy()
    y_remapped[y_remapped == -1] = 0  # Negative
    y_remapped[y_remapped == 1] = 2   # Positive
    # Neutral remains 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], y_remapped, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=y_remapped
    )
    
    # Tokenize for Word2Vec
    tokenized_corpus = [text.split() for text in df['clean_text']]
    
    # Train Word2Vec
    w2v_model = train_word2vec(tokenized_corpus)
    
    # Train TF-IDF
    print("Training TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
    idf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    
    # Create hybrid embeddings for training
    X_train_hybrid = np.array([get_weighted_doc_vector(text, w2v_model, idf_dict) 
                                for text in X_train])
    X_test_hybrid = np.array([get_weighted_doc_vector(text, w2v_model, idf_dict) 
                               for text in X_test])
    
    print(f"Embedding shape: {X_train_hybrid.shape}")
    
    # Build model
    model = build_model(input_dim=X_train_hybrid.shape[1])
    model.summary()
    
    # Compute class weights for imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # Train
    history = model.fit(
        X_train_hybrid, y_train,
        validation_split=VAL_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test_hybrid)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred_classes, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred_classes)
    
    # Save model and artifacts
    print("\nSaving model and artifacts...")
    model.save('models/sentiment_model.h5')
    w2v_model.save('models/word2vec.model')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    
    with open('models/idf_dict.pkl', 'wb') as f:
        pickle.dump(idf_dict, f)
    
    print("✅ Training complete! Model saved to 'models/' directory.")

if __name__ == "__main__":
    main()
