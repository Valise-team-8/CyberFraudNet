import torch

class Config:
    # Raw data paths
    FRAUD_DATA_PATH = "data/raw/Fraud_Data.csv"
    CUSTOMER_DATA_PATH = "data/raw/Customer_DF (1).csv"
    TRANSACTION_DATA_PATH = "data/raw/cust_transaction_details (1).csv"
    FINANCIAL_ANOMALY_PATH = "data/raw/financial_anomaly_data.csv"
    IP_COUNTRY_PATH = "data/raw/IpAddress_to_Country.csv"
    
    # Processed data paths
    PROCESSED_DATA_PATH = "data/processed/combined_fraud_data.csv"
    
    # Model parameters (optimized for RTX 2050 4GB VRAM or CPU)
    LR = 0.001
    HIDDEN_CHANNELS = 32  # Balanced for 4GB VRAM
    BATCH_SIZE = 8 if torch.cuda.is_available() else 16  # Smaller for GPU memory
    EPOCHS = 25  # Balanced training time
    
    # Hardware-specific optimization
    USE_MIXED_PRECISION = torch.cuda.is_available()
    BERT_BATCH_SIZE = 2 if torch.cuda.is_available() else 4  # Very small for GPU VRAM
    
    # System optimizations
    NUM_WORKERS = 2  # Conservative for stability
    PIN_MEMORY = torch.cuda.is_available()
    
    # Memory optimization
    GRADIENT_ACCUMULATION_STEPS = 4 if torch.cuda.is_available() else 1  # Simulate larger batch on GPU
    MAX_GRAD_NORM = 1.0
    EMPTY_CACHE_FREQUENCY = 5  # Clear GPU cache frequently
