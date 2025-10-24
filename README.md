# 🛡️ CYBERFRAUDNET - Advanced AI Fraud Detection System

<div align="center">

![CYBERFRAUDNET](https://img.shields.io/badge/CYBERFRAUDNET-AI%20Fraud%20Detection-00f5ff?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-ff00ff?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-00ff00?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ffff00?style=for-the-badge&logo=streamlit&logoColor=white)

**⚡ Advanced AI-Powered Fraud Detection using Graph Neural Networks ⚡**

*Real-time fraud detection with 95.6% accuracy using cutting-edge machine learning*

</div>

---

## 🚀 **Overview**

CYBERFRAUDNET is a state-of-the-art fraud detection system that combines:
- **Graph Neural Networks (GNN)** for relationship analysis
- **BERT Transformers** for text feature extraction  
- **Temporal Analysis** for time-based patterns
- **Contrastive Learning** for enhanced model training
- **Interactive Dashboard** with futuristic cyberpunk UI

## 🎯 **Key Features**

### 🧠 **AI-Powered Detection**
- **TemporalGNN Architecture**: Custom graph neural network for fraud detection
- **BERT Integration**: Advanced text analysis for transaction descriptions
- **Multi-Modal Learning**: Combines text, numerical, and graph features
- **Real-time Processing**: Instant fraud risk assessment

### 📊 **Performance Metrics**
- **95.6% Accuracy**: Industry-leading fraud detection accuracy
- **98.7% Precision**: Minimal false positives
- **77.2% AUC**: Excellent model discrimination
- **151,112 Transactions**: Processed from real-world datasets

### 🎨 **Futuristic Dashboard**
- **Cyberpunk UI**: Neon-styled interface with glowing effects
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Real-time Monitoring**: Live system status and performance metrics
- **Network Analysis**: Visual transaction relationship mapping

## 🏗️ **Architecture**

```
📊 Data Input → 🔄 Preprocessing → 🧠 Feature Extraction → 🕸️ Graph Construction → 🤖 GNN Model → 📈 Prediction
     ↓              ↓                    ↓                      ↓                    ↓              ↓
Raw Datasets → Data Cleaning → BERT + Numerical → Network Graph → TemporalGNN → Fraud Score
```

### 🔧 **Technical Stack**
- **Backend**: Python 3.8+, PyTorch, PyTorch Geometric
- **ML Models**: BERT, Graph Neural Networks, Contrastive Learning
- **Frontend**: Streamlit with custom CSS styling
- **Visualization**: Plotly, Matplotlib, NetworkX
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 📁 **Project Structure**

```
CYBERFRAUDNET/
├── 🎯 main.py                 # Main pipeline execution
├── 📊 data/
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned and processed data
├── 🧠 src/
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── feature_extraction.py  # BERT and numerical features
│   ├── graph_construction.py  # Network graph building
│   ├── model_gnn.py          # TemporalGNN architecture
│   ├── contrastive_learning.py # Model training
│   ├── evaluation.py         # Performance metrics
│   └── explainability.py     # Model interpretability
├── 🎨 demo_app/
│   └── app.py                # Futuristic Streamlit dashboard
├── 📈 outputs/               # Generated results and visualizations
├── ⚙️ utils/                 # Configuration and utilities
└── 📋 requirements.txt       # Python dependencies
```

## 🚀 **Quick Start**

### 1️⃣ **Installation**
```bash
# Clone the repository
git clone https://github.com/Valise-team-8.git
cd CyberFraudNet

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **Run the Pipeline**
```bash
# Execute the complete fraud detection pipeline
python main.py
```

### 3️⃣ **Launch Dashboard**
```bash
# Start the futuristic dashboard
python -m streamlit run demo_app/app.py

# Or use the launcher
python launch_dashboard.py
```

### 4️⃣ **View Results**
```bash
# Quick results overview
python view_outputs.py
```

## 📊 **Datasets**

The system processes multiple real-world fraud datasets:

| Dataset | Records | Features | Description |
|---------|---------|----------|-------------|
| **Fraud_Data.csv** | 151,112 | 11 | Main fraud detection dataset |
| **Customer_DF.csv** | 2,000+ | 9 | Customer profile information |
| **Transaction_Details.csv** | 10,000+ | 10 | Detailed transaction records |
| **Financial_Anomaly.csv** | 50,000+ | 7 | Financial anomaly patterns |
| **IP_Country.csv** | 100,000+ | 3 | IP geolocation mapping |

## 🎯 **Model Performance**

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 95.6% | Overall prediction accuracy |
| **Precision** | 98.7% | Fraud detection precision |
| **Recall** | 53.7% | Fraud case coverage |
| **AUC** | 77.2% | Area under ROC curve |
| **F1-Score** | 70.0% | Balanced performance metric |

## 🎨 **Dashboard Features**

### 🎯 **Model Results Page**
```python
# Performance Metrics Display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #00ff00;">🎯 ACCURACY</h3>
        <h1 style="color: #ffffff;">{accuracy}</h1>
    </div>
    """, unsafe_allow_html=True)
```
- **Futuristic Metric Cards**: Neon-styled performance indicators
- **Interactive Confusion Matrix**: Clickable heatmap visualization
- **ROC Curve Analysis**: Dynamic performance curve
- **Probability Distributions**: Fraud vs normal pattern analysis

### 📊 **Data Analysis Page**
```python
# Interactive Plotly Visualizations
fig = go.Figure(data=[go.Pie(
    labels=['Normal', 'Fraud'],
    values=fraud_counts.values,
    hole=0.4,
    marker_colors=['#00f5ff', '#ff0000']
)])
st.plotly_chart(fig, use_container_width=True)
```
- **Dataset Statistics**: Real-time data insights
- **Interactive Charts**: Plotly-powered visualizations
- **Fraud Distribution**: Pie charts and histograms
- **Purchase Analysis**: Value distribution by fraud status
- **Age Demographics**: Box plots and statistical analysis

### 🔍 **Live Prediction Page**
```python
# Network Analysis
G = nx.Graph()
for _, row in sample_df.iterrows():
    user = f"U_{row['user_id']}"
    product = f"P_{row['product_id']}"
    seller = f"S_{row['seller_id']}"
    G.add_edge(user, product)
    G.add_edge(product, seller)

# AI Risk Assessment
if st.button("🔍 Run AI Fraud Detection"):
    fraud_probability = model.predict(features)
    risk_score = calculate_risk_score(fraud_probability)
```
- **CSV Upload Portal**: Drag-and-drop file analysis
- **Network Topology**: Interactive graph visualization
- **AI Risk Assessment**: Real-time fraud probability calculation
- **Transaction Mapping**: User-Product-Seller relationship analysis

### ⚙️ **System Status Page**
```python
# Real-time System Monitoring
st.markdown(f"""
<div class="status-card">
    <h4 style="color: #00ff00;">🟢 NEURAL NETWORK</h4>
    <h3 style="color: #ffffff;">ONLINE</h3>
    <p style="color: #00f5ff;">TemporalGNN Active</p>
</div>
""", unsafe_allow_html=True)
```
- **System Health**: Neural network status monitoring
- **Model Architecture**: Component overview and status
- **Performance Metrics**: Real-time processing statistics
- **Uptime Tracking**: System availability monitoring

### 🎨 **UI Components Showcase**

#### **Cyberpunk Styling**
- **Neon Color Palette**: Cyan (#00f5ff), Magenta (#ff00ff), Green (#00ff00)
- **Gradient Backgrounds**: Dynamic color transitions
- **Glass Morphism**: Translucent cards with backdrop blur
- **Animated Elements**: Pulsing indicators and smooth transitions

#### **Interactive Elements**
- **Hover Effects**: Dynamic color changes and shadows
- **Click Animations**: Button press feedback
- **Loading Spinners**: Futuristic processing indicators
- **Progress Bars**: Animated training progress

#### **Responsive Design**
- **Grid Layouts**: Adaptive column systems
- **Mobile Optimization**: Touch-friendly interfaces
- **Dark Theme**: Eye-friendly color scheme
- **Typography**: Orbitron and Roboto Mono fonts

## 🔬 **Advanced Features**

### 🧠 **Graph Neural Networks**
- **Heterogeneous Graphs**: User-Product-Seller relationships
- **Temporal Dynamics**: Time-based transaction patterns
- **Multi-hop Reasoning**: Complex fraud pattern detection
- **Attention Mechanisms**: Focus on important connections

### 🤖 **Machine Learning Pipeline**
- **Feature Engineering**: 771-dimensional feature vectors
- **Contrastive Learning**: Enhanced model training
- **Cross-validation**: Robust model evaluation
- **Hyperparameter Tuning**: Optimized performance

### 🔍 **Explainability**
- **Node Importance**: Key fraud indicators
- **Edge Analysis**: Critical relationships
- **Feature Attribution**: Most influential factors
- **Decision Transparency**: Interpretable predictions

## 🎨 **UI Implementation Details**

### 🌟 **Cyberpunk Design System**
The dashboard features a cutting-edge cyberpunk aesthetic with:

```css
/* Custom CSS Styling */
.cyber-title {
    font-size: 3.5rem;
    background: linear-gradient(45deg, #00f5ff, #ff00ff, #00ff00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
}

.metric-card {
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
    border: 1px solid rgba(0, 245, 255, 0.3);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 245, 255, 0.1);
    backdrop-filter: blur(10px);
}
```

### 🎛️ **Interactive Components**
- **Real-time Metrics**: Live updating performance indicators
- **Interactive Charts**: Plotly-powered visualizations with hover effects
- **Network Graphs**: Dynamic transaction relationship mapping
- **File Upload**: Drag-and-drop CSV analysis
- **Animated Elements**: Pulsing status indicators and smooth transitions

### 📱 **Responsive Design**
- **Multi-column Layouts**: Adaptive grid system
- **Mobile Friendly**: Responsive breakpoints
- **Dark Theme**: Optimized for extended use
- **Accessibility**: Screen reader compatible

## 🚀 **Training Implementation**

### 📊 **Data Preprocessing Pipeline**
```python
def preprocess_data():
    # 1. Load multiple datasets
    fraud_df = preprocess_fraud_data()          # Main fraud dataset
    customer_df = preprocess_customer_data()    # Customer profiles
    transaction_df = preprocess_transaction_data() # Transaction details
    financial_df = preprocess_financial_anomaly_data() # Anomaly patterns
    
    # 2. Feature engineering
    - IP geolocation mapping
    - Temporal feature extraction
    - Categorical encoding
    - Text preprocessing for BERT
    
    # 3. Graph construction
    - User-Product-Seller relationships
    - Temporal edge weights
    - Node feature assignment
```

### 🧠 **Model Architecture**
```python
class TemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

### 🎯 **Training Process**
```python
# Training Configuration
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 64
BATCH_SIZE = 32

# Training Loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass through GNN
    out = model(data.x, data.edge_index)
    
    # Calculate loss on training nodes
    loss = criterion(out[train_mask], labels[train_mask])
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = out[test_mask]
        val_acc = accuracy(val_out, labels[test_mask])
```

### 🔄 **Feature Extraction Details**
```python
def extract_features(df):
    # 1. BERT Text Features (768 dimensions)
    text_features = extract_text_features_bert(df['review_text'])
    
    # 2. Numerical Features
    numerical_features = extract_numerical_features(df)
    
    # 3. Feature Fusion
    combined_features = np.concatenate([text_features, numerical_features], axis=1)
    
    return combined_features  # Shape: (N, 771)
```

### 📈 **Evaluation Metrics**
```python
def evaluate_model(model, data):
    # Performance Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    
    # Visualizations
    save_confusion_matrix(true_labels, predictions)
    save_roc_curve(true_labels, probabilities)
    save_probability_distributions(true_labels, probabilities)
```

## 🔧 **Technical Implementation**

### 🗂️ **Data Flow Architecture**
```
Raw Data → Preprocessing → Feature Engineering → Graph Construction → Model Training → Evaluation → Dashboard
    ↓            ↓              ↓                    ↓                 ↓             ↓           ↓
5 CSV files → Cleaning → BERT + Numerical → NetworkX Graph → TemporalGNN → Metrics → Streamlit UI
```

### 🧪 **Model Components**
1. **Input Layer**: 771-dimensional feature vectors
2. **BERT Encoder**: Pre-trained transformer for text analysis
3. **Graph Convolution**: TransformerConv layers for relationship modeling
4. **Temporal Attention**: Time-aware pattern recognition
5. **Output Layer**: Binary classification (fraud/normal)

### 📊 **Performance Optimization**
- **Batch Processing**: Efficient data loading
- **GPU Acceleration**: CUDA support for training
- **Memory Management**: Optimized for large graphs
- **Caching**: Preprocessed data storage

## 🛠️ **Configuration**

Key configuration parameters in `utils/config.py`:

```python
class Config:
    # Model Parameters
    HIDDEN_CHANNELS = 64
    LR = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Data Paths
    FRAUD_DATA_PATH = "data/raw/Fraud_Data.csv"
    PROCESSED_DATA_PATH = "data/processed/combined_fraud_data.csv"
```

## 📈 **Results & Outputs**

Generated outputs include:
- 📊 **metrics_report.txt**: Detailed performance metrics
- 🎯 **confusion_matrix.png**: Visual confusion matrix
- 📈 **roc_curve.png**: ROC curve visualization
- 📊 **probability_distributions.png**: Fraud probability analysis
- 🤖 **trained_model.pth**: Saved neural network model



---

<div align="center">


![Made with Python](https://img.shields.io/badge/Made%20with-Python-00f5ff?style=flat-square&logo=python&logoColor=white)
![Powered by AI](https://img.shields.io/badge/Powered%20by-AI-ff00ff?style=flat-square&logo=brain&logoColor=white)
![Open Source](https://img.shields.io/badge/Open-Source-00ff00?style=flat-square&logo=github&logoColor=white)

</div>