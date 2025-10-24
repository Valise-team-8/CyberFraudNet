import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from utils.logger import log

# Initialize BERT model for text feature extraction
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    BERT_AVAILABLE = True
except:
    log("⚠️ BERT model not available, using alternative text features")
    BERT_AVAILABLE = False

def extract_text_features_bert(texts):
    """Extract BERT embeddings from text"""
    text_features = []
    for text in tqdm(texts, desc="Extracting BERT features"):
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        text_features.append(emb)
    return np.array(text_features)

def extract_text_features_simple(texts):
    """Simple text feature extraction as fallback"""
    features = []
    for text in texts:
        text_str = str(text).lower()
        # Simple text statistics
        feature_vec = [
            len(text_str),  # text length
            text_str.count(' '),  # word count
            text_str.count('fraud'),  # fraud mentions
            text_str.count('fake'),  # fake mentions
            text_str.count('good'),  # positive words
            text_str.count('bad'),   # negative words
        ]
        # Pad to match BERT dimension (768)
        feature_vec.extend([0] * (768 - len(feature_vec)))
        features.append(feature_vec[:768])
    return np.array(features)

def extract_numerical_features(df):
    """Extract numerical features from the dataframe"""
    log("Extracting numerical features...")
    
    numerical_cols = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in ['user_id', 'product_id', 'seller_id', 'is_fraud']:
            numerical_cols.append(col)
    
    if numerical_cols:
        numerical_features = df[numerical_cols].fillna(0).values
        log(f"Extracted {len(numerical_cols)} numerical features: {numerical_cols}")
    else:
        # Create basic numerical features if none exist
        numerical_features = np.random.randn(len(df), 10)  # 10 synthetic features
        log("Created synthetic numerical features")
    
    return numerical_features

def extract_features(df):
    """Extract comprehensive features combining text and numerical data"""
    log("🔍 Starting feature extraction...")
    
    # Extract text features
    if 'review_text' in df.columns:
        if BERT_AVAILABLE:
            text_features = extract_text_features_bert(df['review_text'])
        else:
            text_features = extract_text_features_simple(df['review_text'])
        log(f"Text features shape: {text_features.shape}")
    else:
        # Create dummy text features if no text column
        text_features = np.random.randn(len(df), 768)
        log("Created synthetic text features")
    
    # Extract numerical features
    numerical_features = extract_numerical_features(df)
    
    # Combine features
    if numerical_features.shape[1] > 0:
        # Normalize numerical features to match text feature scale
        numerical_features = (numerical_features - numerical_features.mean(axis=0)) / (numerical_features.std(axis=0) + 1e-8)
        combined_features = np.concatenate([text_features, numerical_features], axis=1)
        log(f"Combined features shape: {combined_features.shape}")
    else:
        combined_features = text_features
    
    log("✅ Feature extraction complete")
    return combined_features
