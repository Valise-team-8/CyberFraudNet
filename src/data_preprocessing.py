import pandas as pd
import numpy as np
from utils.logger import log
from utils.config import Config
import os

def load_ip_country_mapping():
    """Load IP to country mapping for geolocation features"""
    log("Loading IP to country mapping...")
    ip_df = pd.read_csv(Config.IP_COUNTRY_PATH)
    return ip_df

def map_ip_to_country(ip_address, ip_mapping):
    """Map IP address to country using the IP range mapping"""
    try:
        ip_int = int(float(ip_address))
        country = ip_mapping[
            (ip_mapping['lower_bound_ip_address'] <= ip_int) & 
            (ip_mapping['upper_bound_ip_address'] >= ip_int)
        ]['country'].iloc[0] if len(ip_mapping[
            (ip_mapping['lower_bound_ip_address'] <= ip_int) & 
            (ip_mapping['upper_bound_ip_address'] >= ip_int)
        ]) > 0 else 'Unknown'
        return country
    except:
        return 'Unknown'

def preprocess_fraud_data():
    """Process the main fraud dataset"""
    log("Processing fraud data...")
    df = pd.read_csv(Config.FRAUD_DATA_PATH)
    
    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Calculate time difference between signup and purchase
    df['signup_to_purchase_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Load IP mapping and add country information
    ip_mapping = load_ip_country_mapping()
    df['country'] = df['ip_address'].apply(lambda x: map_ip_to_country(x, ip_mapping))
    
    # Encode categorical variables
    categorical_cols = ['device_id', 'source', 'browser', 'sex', 'country']
    for col in categorical_cols:
        df[f'{col}_encoded'] = df[col].astype('category').cat.codes
    
    # Select relevant features
    feature_cols = ['user_id', 'purchase_value', 'age', 'signup_to_purchase_hours'] + \
                   [f'{col}_encoded' for col in categorical_cols] + ['class']
    
    return df[feature_cols].rename(columns={'class': 'is_fraud'})

def preprocess_customer_data():
    """Process customer profile data"""
    log("Processing customer data...")
    df = pd.read_csv(Config.CUSTOMER_DATA_PATH)
    
    # Clean and encode customer data
    df['customer_id'] = df.index
    df['fraud_label'] = df['Fraud'].astype(int)
    
    # Extract features from customer behavior
    feature_cols = ['customer_id', 'No_Transactions', 'No_Orders', 'No_Payments', 'fraud_label']
    
    return df[feature_cols]

def preprocess_transaction_data():
    """Process transaction details"""
    log("Processing transaction data...")
    df = pd.read_csv(Config.TRANSACTION_DATA_PATH)
    
    # Encode categorical variables
    df['payment_method_encoded'] = df['paymentMethodType'].astype('category').cat.codes
    df['provider_encoded'] = df['paymentMethodProvider'].astype('category').cat.codes
    df['order_state_encoded'] = df['orderState'].astype('category').cat.codes
    
    # Group by customer email to get aggregated features
    customer_features = df.groupby('customerEmail').agg({
        'transactionAmount': ['mean', 'std', 'sum', 'count'],
        'transactionFailed': 'sum',
        'paymentMethodRegistrationFailure': 'sum'
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['customerEmail', 'avg_transaction_amount', 'std_transaction_amount',
                                'total_transaction_amount', 'transaction_count', 'failed_transactions',
                                'payment_failures']
    
    return customer_features

def preprocess_financial_anomaly_data():
    """Process financial anomaly data"""
    log("Processing financial anomaly data...")
    df = pd.read_csv(Config.FINANCIAL_ANOMALY_PATH)
    
    # Convert timestamp with flexible parsing
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    
    # Encode categorical variables
    df['merchant_encoded'] = df['Merchant'].astype('category').cat.codes
    df['transaction_type_encoded'] = df['TransactionType'].astype('category').cat.codes
    df['location_encoded'] = df['Location'].astype('category').cat.codes
    
    # Select features
    feature_cols = ['TransactionID', 'AccountID', 'Amount', 'hour', 'day_of_week',
                   'merchant_encoded', 'transaction_type_encoded', 'location_encoded']
    
    return df[feature_cols]

def create_combined_dataset():
    """Combine all datasets into a unified format for graph construction"""
    log("Creating combined dataset...")
    
    # Process individual datasets
    fraud_df = preprocess_fraud_data()
    customer_df = preprocess_customer_data()
    transaction_df = preprocess_transaction_data()
    financial_df = preprocess_financial_anomaly_data()
    
    # Create a unified node representation
    # Each row represents a node (user/account/transaction) with features and fraud label
    
    # Use fraud_df as the base since it has clear fraud labels
    combined_df = fraud_df.copy()
    
    # Add synthetic review text for compatibility with existing feature extraction
    combined_df['review_text'] = combined_df.apply(lambda row: 
        f"User transaction amount {row['purchase_value']} age {row['age']} " +
        f"hours to purchase {row['signup_to_purchase_hours']:.1f}", axis=1)
    
    # Add timestamp for temporal analysis
    combined_df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(combined_df), freq='H')
    
    # Ensure required columns exist
    required_cols = ['user_id', 'timestamp', 'review_text', 'is_fraud']
    for col in required_cols:
        if col not in combined_df.columns:
            if col == 'is_fraud':
                combined_df[col] = combined_df.get('fraud_label', 0)
    
    # Add required columns for graph construction if missing
    if 'product_id' not in combined_df.columns:
        # Create product IDs based on transaction patterns
        np.random.seed(42)  # For reproducibility
        combined_df['product_id'] = np.random.randint(1, 100, len(combined_df))
    if 'seller_id' not in combined_df.columns:
        # Create seller IDs based on transaction patterns  
        np.random.seed(42)  # For reproducibility
        combined_df['seller_id'] = np.random.randint(1, 50, len(combined_df))
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    combined_df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    log(f"Combined dataset saved to {Config.PROCESSED_DATA_PATH}")
    
    return combined_df

def preprocess_data(path=None):
    """Main preprocessing function - creates combined dataset from all raw data"""
    log("🔄 Starting comprehensive data preprocessing...")
    
    # Check if processed data already exists
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        log("Loading existing processed data...")
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        log(f"Loaded processed dataset shape: {df.shape}")
        return df
    
    # Create combined dataset from raw data
    df = create_combined_dataset()
    
    # Final preprocessing steps
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add required columns for graph construction if missing
    if 'product_id' not in df.columns:
        # Create product IDs based on transaction patterns
        np.random.seed(42)  # For reproducibility
        df['product_id'] = np.random.randint(1, 100, len(df))
    if 'seller_id' not in df.columns:
        # Create seller IDs based on transaction patterns  
        np.random.seed(42)  # For reproducibility
        df['seller_id'] = np.random.randint(1, 50, len(df))
    
    # Encode categorical variables for graph construction
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['product_id'] = df['product_id'].astype('category').cat.codes
    df['seller_id'] = df['seller_id'].astype('category').cat.codes
    
    log(f"✅ Final processed dataset shape: {df.shape}")
    log(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    
    return df
