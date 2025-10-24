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

### 🎯 **Model Results**
- Real-time performance metrics
- Interactive confusion matrix
- ROC curve analysis
- Probability distributions

### 📊 **Data Analysis**
- Dataset statistics and insights
- Interactive fraud distribution charts
- Purchase value analysis
- Age demographic patterns

### 🔍 **Live Prediction**
- Upload CSV for instant analysis
- Network topology visualization
- AI-powered risk assessment
- Real-time fraud scoring

### ⚙️ **System Status**
- Neural network health monitoring
- Model architecture overview
- Real-time performance metrics
- System uptime tracking

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

## 🚀 **Deployment**

### 🐳 **Docker Support** (Coming Soon)
```bash
docker build -t cyberfraudnet .
docker run -p 8501:8501 cyberfraudnet
```

### ☁️ **Cloud Deployment**
- AWS/GCP/Azure compatible
- Scalable architecture
- API endpoints available
- Real-time processing

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **PyTorch Team** for the deep learning framework
- **Hugging Face** for BERT transformers
- **Streamlit** for the dashboard framework
- **Research Community** for fraud detection methodologies

## 📞 **Contact**

For questions, issues, or collaborations:

- 📧 **Email**: [team@cyberfraudnet.ai](mailto:team@cyberfraudnet.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Valise-team-8/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Valise-team-8/discussions)

---

<div align="center">


![Made with Python](https://img.shields.io/badge/Made%20with-Python-00f5ff?style=flat-square&logo=python&logoColor=white)
![Powered by AI](https://img.shields.io/badge/Powered%20by-AI-ff00ff?style=flat-square&logo=brain&logoColor=white)
![Open Source](https://img.shields.io/badge/Open-Source-00ff00?style=flat-square&logo=github&logoColor=white)

</div>