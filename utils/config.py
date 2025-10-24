class Config:
    # Raw data paths
    FRAUD_DATA_PATH = "data/raw/Fraud_Data.csv"
    CUSTOMER_DATA_PATH = "data/raw/Customer_DF (1).csv"
    TRANSACTION_DATA_PATH = "data/raw/cust_transaction_details (1).csv"
    FINANCIAL_ANOMALY_PATH = "data/raw/financial_anomaly_data.csv"
    IP_COUNTRY_PATH = "data/raw/IpAddress_to_Country.csv"
    
    # Processed data paths
    PROCESSED_DATA_PATH = "data/processed/combined_fraud_data.csv"
    
    # Model parameters
    LR = 0.001
    HIDDEN_CHANNELS = 64
    BATCH_SIZE = 32
    EPOCHS = 100
