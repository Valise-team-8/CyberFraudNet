import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from utils.logger import log

# Initialize BERT model for text feature extraction with GPU support
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(device)
    BERT_AVAILABLE = True
    log(f"🤖 BERT model loaded on {device}")
except:
    log("⚠️ BERT model not available, using alternative text features")
    BERT_AVAILABLE = False
    device = torch.device('cpu')

def extract_text_features_bert(texts):
    """Extract BERT embeddings from text using GPU acceleration"""
    text_features = []
    batch_size = 32 if device.type == 'cuda' else 8  # Larger batches for GPU
    
    # Process in batches for better GPU utilization
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting BERT features on {device}"):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = tokenizer(
            [str(text) for text in batch_texts], 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Move inputs to GPU
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**batch_inputs)
            # Get mean pooled embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Move back to CPU for numpy conversion
            embeddings = embeddings.cpu().numpy()
            text_features.extend(embeddings)
    
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
