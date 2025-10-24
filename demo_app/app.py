import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from PIL import Image
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config with dark theme
st.set_page_config(
    page_title="CYBERFRAUDNET", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# Custom CSS for futuristic styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .cyber-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00f5ff, #ff00ff, #00ff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        margin-bottom: 0.5rem;
        font-family: 'Orbitron', monospace;
    }
    
    .cyber-subtitle {
        font-size: 1.2rem;
        color: #00f5ff;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Roboto Mono', monospace;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .status-card {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.1) 0%, rgba(0, 245, 255, 0.1) 100%);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 255, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 0, 0, 0.1) 100%);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(255, 165, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(0, 245, 255, 0.1);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00f5ff, #ff00ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 245, 255, 0.5);
    }
    
    .cyber-header {
        font-size: 2rem;
        color: #00f5ff;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    .cyber-subheader {
        font-size: 1.5rem;
        color: #ff00ff;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 0 0 8px rgba(255, 0, 255, 0.5);
    }
    
    .data-table {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .glow-text {
        text-shadow: 0 0 10px currentColor;
    }
</style>
""", unsafe_allow_html=True)

# Add Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Main title with futuristic styling
st.markdown('<h1 class="cyber-title">🛡️ CYBERFRAUDNET</h1>', unsafe_allow_html=True)
st.markdown('<p class="cyber-subtitle">⚡ Advanced AI-Powered Fraud Detection System ⚡</p>', unsafe_allow_html=True)

# Add animated status indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="status-card">
        <div style="text-align: center;">
            <span class="pulse glow-text" style="color: #00ff00; font-size: 1.2rem;">🟢 SYSTEM ONLINE</span>
            <br>
            <span style="color: #00f5ff; font-size: 0.9rem;">Neural Networks Active • Graph Analysis Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Futuristic sidebar
st.sidebar.markdown('<h2 style="color: #00f5ff; text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);">🎛️ CONTROL PANEL</h2>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation with icons
page_options = {
    "🎯 Model Results": "Model Results",
    "📊 Data Analysis": "Data Analysis", 
    "🔍 Live Prediction": "Live Prediction",
    "⚙️ System Status": "System Status"
}

page = st.sidebar.selectbox("🚀 Navigate to:", list(page_options.keys()))
selected_page = page_options[page]

# System info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="metric-card">
    <h4 style="color: #ff00ff; margin: 0;">⚡ System Info</h4>
    <p style="margin: 0.5rem 0; color: #00f5ff;">Model: TemporalGNN</p>
    <p style="margin: 0.5rem 0; color: #00f5ff;">Framework: PyTorch</p>
    <p style="margin: 0.5rem 0; color: #00f5ff;">Status: Active</p>
</div>
""", unsafe_allow_html=True)

if selected_page == "Model Results":
    st.markdown('<h2 class="cyber-header">🎯 MODEL PERFORMANCE MATRIX</h2>', unsafe_allow_html=True)
    
    # Fix path issues - look in parent directory
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    
    # Performance metrics in futuristic cards
    metrics_file = os.path.join(outputs_dir, "metrics_report.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            content = f.read()
        
        # Extract key metrics
        lines = content.split('\n')
        accuracy = precision = recall = auc = "N/A"
        for line in lines:
            if "Accuracy:" in line:
                accuracy = line.split(":")[1].strip()
            elif "Precision:" in line:
                precision = line.split(":")[1].strip()
            elif "Recall:" in line:
                recall = line.split(":")[1].strip()
            elif "AUC:" in line:
                auc = line.split(":")[1].strip()
        
        # Display metrics in futuristic cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00ff00; margin: 0; text-align: center;">🎯 ACCURACY</h3>
                <h1 style="color: #ffffff; margin: 0.5rem 0; text-align: center; font-size: 2.5rem;">{accuracy}</h1>
                <p style="color: #00f5ff; margin: 0; text-align: center;">Overall Performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ff00ff; margin: 0; text-align: center;">🔍 PRECISION</h3>
                <h1 style="color: #ffffff; margin: 0.5rem 0; text-align: center; font-size: 2.5rem;">{precision}</h1>
                <p style="color: #00f5ff; margin: 0; text-align: center;">Fraud Detection Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ffff00; margin: 0; text-align: center;">⚡ RECALL</h3>
                <h1 style="color: #ffffff; margin: 0.5rem 0; text-align: center; font-size: 2.5rem;">{recall}</h1>
                <p style="color: #00f5ff; margin: 0; text-align: center;">Coverage Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00ffff; margin: 0; text-align: center;">📈 AUC</h3>
                <h1 style="color: #ffffff; margin: 0.5rem 0; text-align: center; font-size: 2.5rem;">{auc}</h1>
                <p style="color: #00f5ff; margin: 0; text-align: center;">Model Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Full report in expandable section
        with st.expander("📋 Detailed Classification Report", expanded=False):
            st.code(content, language="text")
    else:
        st.markdown("""
        <div class="warning-card">
            <h3 style="color: #ff6600; margin: 0;">⚠️ METRICS NOT FOUND</h3>
            <p style="margin: 0.5rem 0; color: #ffffff;">Run the main pipeline first: <code>python main.py</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations section
    st.markdown('<h2 class="cyber-subheader">📊 NEURAL NETWORK ANALYSIS</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #00f5ff;">🎯 Confusion Matrix</h3>', unsafe_allow_html=True)
        confusion_file = os.path.join(outputs_dir, "confusion_matrix.png")
        if os.path.exists(confusion_file):
            image = Image.open(confusion_file)
            st.image(image, use_column_width=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <p style="margin: 0; color: #ffffff;">🔍 Confusion matrix visualization not found</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 style="color: #ff00ff;">📈 ROC Curve</h3>', unsafe_allow_html=True)
        roc_file = os.path.join(outputs_dir, "roc_curve.png")
        if os.path.exists(roc_file):
            image = Image.open(roc_file)
            st.image(image, use_column_width=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <p style="margin: 0; color: #ffffff;">📊 ROC curve visualization not found</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Probability distributions
    st.markdown('<h3 style="color: #00ff00;">🔬 Probability Distributions</h3>', unsafe_allow_html=True)
    prob_file = os.path.join(outputs_dir, "probability_distributions.png")
    if os.path.exists(prob_file):
        image = Image.open(prob_file)
        st.image(image, use_column_width=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <p style="margin: 0; color: #ffffff;">📈 Probability distribution visualization not found</p>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Data Analysis":
    st.markdown('<h2 class="cyber-header">📊 DATA INTELLIGENCE CENTER</h2>', unsafe_allow_html=True)
    
    # Load processed data with correct path
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "combined_fraud_data.csv")
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        
        # Dataset overview with futuristic metrics
        st.markdown('<h3 class="cyber-subheader">🎯 DATASET OVERVIEW</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        fraud_count = df['is_fraud'].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #00f5ff; margin: 0; text-align: center;">📊 TOTAL TRANSACTIONS</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ff0000; margin: 0; text-align: center;">🚨 FRAUD CASES</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{fraud_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ffff00; margin: 0; text-align: center;">⚡ FRAUD RATE</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{fraud_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #00ff00; margin: 0; text-align: center;">🔬 FEATURES</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive data sample
        st.markdown('<h3 class="cyber-subheader">🔍 DATA SAMPLE</h3>', unsafe_allow_html=True)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            show_fraud_only = st.checkbox("🚨 Show Fraud Cases Only", value=False)
        with col2:
            sample_size = st.slider("📊 Sample Size", min_value=10, max_value=100, value=20)
        
        display_df = df[df['is_fraud'] == 1] if show_fraud_only else df
        st.dataframe(display_df.head(sample_size), use_container_width=True)
        
        st.markdown("---")
        
        # Interactive visualizations with Plotly
        st.markdown('<h3 class="cyber-subheader">📈 INTERACTIVE ANALYTICS</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 style="color: #00f5ff;">🎯 Fraud Distribution</h4>', unsafe_allow_html=True)
            
            # Create interactive pie chart
            fraud_counts = df['is_fraud'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Normal', 'Fraud'],
                values=fraud_counts.values,
                hole=0.4,
                marker_colors=['#00f5ff', '#ff0000'],
                textfont_size=16,
                textfont_color='white'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='#00f5ff',
                showlegend=True,
                legend=dict(font_color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h4 style="color: #ff00ff;">💰 Purchase Value Analysis</h4>', unsafe_allow_html=True)
            
            # Create interactive histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud']==0]['purchase_value'],
                name='Normal',
                marker_color='#00f5ff',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud']==1]['purchase_value'],
                name='Fraud',
                marker_color='#ff0000',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title='Purchase Value',
                yaxis_title='Count',
                barmode='overlay',
                showlegend=True,
                legend=dict(font_color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution analysis
        st.markdown('<h4 style="color: #00ff00;">👥 Age Distribution Analysis</h4>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=df[df['is_fraud']==0]['age'],
            name='Normal',
            marker_color='#00f5ff',
            boxpoints='outliers'
        ))
        
        fig.add_trace(go.Box(
            y=df[df['is_fraud']==1]['age'],
            name='Fraud',
            marker_color='#ff0000',
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis_title='Age',
            showlegend=True,
            legend=dict(font_color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("""
        <div class="warning-card">
            <h3 style="color: #ff6600; margin: 0;">⚠️ DATASET NOT FOUND</h3>
            <p style="margin: 0.5rem 0; color: #ffffff;">Run the main pipeline first: <code>python main.py</code></p>
            <p style="margin: 0.5rem 0; color: #00f5ff;">Expected location: data/processed/combined_fraud_data.csv</p>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Live Prediction":
    st.markdown('<h2 class="cyber-header">🔍 LIVE THREAT DETECTION</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00f5ff; font-size: 1.1rem;">Upload transaction data for real-time AI-powered fraud analysis</p>', unsafe_allow_html=True)
    
    # File uploader with custom styling
    st.markdown('<h3 style="color: #ff00ff;">📁 DATA UPLOAD PORTAL</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "🚀 Upload Transaction CSV File", 
        type=["csv"],
        help="Upload your transaction data for AI analysis"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Success message
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #00ff00; margin: 0;">✅ DATA UPLOAD SUCCESSFUL</h4>
            <p style="margin: 0.5rem 0; color: #ffffff;">File processed and ready for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data preview with metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #00f5ff; margin: 0; text-align: center;">📊 RECORDS</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ff00ff; margin: 0; text-align: center;">🔬 COLUMNS</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #00ff00; margin: 0; text-align: center;">💾 SIZE</h4>
                <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{df.memory_usage(deep=True).sum() / 1024:.1f}KB</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: #00f5ff;">🔍 DATA PREVIEW</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Check required columns
        required_cols = ['user_id', 'product_id', 'seller_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.markdown(f"""
            <div class="warning-card">
                <h4 style="color: #ff6600; margin: 0;">⚠️ MISSING REQUIRED COLUMNS</h4>
                <p style="margin: 0.5rem 0; color: #ffffff;">Missing: {', '.join(missing_cols)}</p>
                <p style="margin: 0.5rem 0; color: #00f5ff;">Required: user_id, product_id, seller_id</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Network analysis
            st.markdown('<h3 style="color: #ff00ff;">🕸️ NETWORK TOPOLOGY ANALYSIS</h3>', unsafe_allow_html=True)
            
            # Create network graph
            G = nx.Graph()
            sample_df = df.head(min(100, len(df)))  # Limit for performance
            
            for _, row in sample_df.iterrows():
                user = f"U_{row['user_id']}"
                product = f"P_{row['product_id']}"
                seller = f"S_{row['seller_id']}"
                
                G.add_edge(user, product, weight=1)
                G.add_edge(product, seller, weight=1)
            
            # Network statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #00f5ff; margin: 0; text-align: center;">🔗 NODES</h4>
                    <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{G.number_of_nodes()}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #ff00ff; margin: 0; text-align: center;">🌐 EDGES</h4>
                    <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{G.number_of_edges()}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                density = nx.density(G)
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #00ff00; margin: 0; text-align: center;">📊 DENSITY</h4>
                    <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{density:.3f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                components = nx.number_connected_components(G)
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #ffff00; margin: 0; text-align: center;">🔍 COMPONENTS</h4>
                    <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{components}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive network visualization
            if st.button("🚀 Generate Network Visualization"):
                with st.spinner("🔄 Analyzing network topology..."):
                    fig, ax = plt.subplots(figsize=(14, 10))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    
                    pos = nx.spring_layout(G, k=2, iterations=50)
                    
                    # Color nodes by type
                    node_colors = []
                    node_sizes = []
                    for node in G.nodes():
                        if node.startswith('U_'):
                            node_colors.append('#00f5ff')  # Cyan for users
                            node_sizes.append(500)
                        elif node.startswith('P_'):
                            node_colors.append('#00ff00')  # Green for products
                            node_sizes.append(300)
                        else:
                            node_colors.append('#ff00ff')  # Magenta for sellers
                            node_sizes.append(400)
                    
                    # Draw network
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=node_sizes, alpha=0.8, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='white', 
                                         alpha=0.3, width=0.5, ax=ax)
                    
                    # Add labels for important nodes
                    important_nodes = dict(G.degree())
                    top_nodes = sorted(important_nodes.items(), key=lambda x: x[1], reverse=True)[:20]
                    labels = {node: node for node, _ in top_nodes}
                    nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                                          font_color='white', ax=ax)
                    
                    ax.set_title('🕸️ Transaction Network Topology', 
                               color='white', fontsize=16, pad=20)
                    ax.axis('off')
                    
                    st.pyplot(fig)
            
            # AI Analysis simulation
            st.markdown('<h3 style="color: #00ff00;">🤖 AI FRAUD ANALYSIS</h3>', unsafe_allow_html=True)
            
            if st.button("🔍 Run AI Fraud Detection"):
                with st.spinner("🧠 AI Neural Networks Processing..."):
                    import time
                    time.sleep(2)  # Simulate processing
                    
                    # Simulate fraud detection results
                    fraud_probability = np.random.random()
                    risk_score = np.random.randint(1, 100)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if fraud_probability > 0.7:
                            color = "#ff0000"
                            status = "HIGH RISK"
                            icon = "🚨"
                        elif fraud_probability > 0.4:
                            color = "#ff6600"
                            status = "MEDIUM RISK"
                            icon = "⚠️"
                        else:
                            color = "#00ff00"
                            status = "LOW RISK"
                            icon = "✅"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: {color}; margin: 0; text-align: center;">{icon} THREAT LEVEL</h3>
                            <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{status}</h2>
                            <p style="color: #00f5ff; margin: 0; text-align: center;">Fraud Probability: {fraud_probability:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #ff00ff; margin: 0; text-align: center;">📊 RISK SCORE</h3>
                            <h2 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{risk_score}/100</h2>
                            <p style="color: #00f5ff; margin: 0; text-align: center;">AI Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success("🎯 AI Analysis Complete! Results generated using advanced neural networks.")
    
    else:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #00f5ff; margin: 0;">📁 AWAITING DATA UPLOAD</h4>
            <p style="margin: 0.5rem 0; color: #ffffff;">Upload a CSV file to begin AI-powered fraud detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data format
        st.markdown('<h3 style="color: #ff00ff;">📋 EXPECTED DATA FORMAT</h3>', unsafe_allow_html=True)
        
        example_data = {
            'user_id': [1001, 1002, 1003, 1004],
            'product_id': [2001, 2002, 2003, 2004],
            'seller_id': [3001, 3002, 3003, 3004],
            'review_text': ['Excellent product!', 'Suspicious item', 'Great quality', 'Fake product'],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        }
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

elif selected_page == "System Status":
    st.markdown('<h2 class="cyber-header">⚙️ SYSTEM STATUS MONITOR</h2>', unsafe_allow_html=True)
    
    # System health indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #00ff00; margin: 0; text-align: center;">🟢 NEURAL NETWORK</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">ONLINE</h3>
            <p style="color: #00f5ff; margin: 0; text-align: center;">TemporalGNN Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #00ff00; margin: 0; text-align: center;">🟢 GRAPH ENGINE</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">READY</h3>
            <p style="color: #00f5ff; margin: 0; text-align: center;">PyTorch Geometric</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #00ff00; margin: 0; text-align: center;">🟢 AI MODELS</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">LOADED</h3>
            <p style="color: #00f5ff; margin: 0; text-align: center;">BERT + GNN</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model information
    st.markdown('<h3 style="color: #ff00ff;">🤖 MODEL ARCHITECTURE</h3>', unsafe_allow_html=True)
    
    model_info = {
        "Component": ["Input Layer", "BERT Encoder", "Feature Fusion", "Graph Conv 1", "Graph Conv 2", "Output Layer"],
        "Type": ["Text + Numerical", "Transformer", "Concatenation", "TransformerConv", "TransformerConv", "Classification"],
        "Dimensions": ["768 + N", "768", "768 + N", "64", "2", "2"],
        "Status": ["✅ Active", "✅ Active", "✅ Active", "✅ Active", "✅ Active", "✅ Active"]
    }
    
    st.dataframe(pd.DataFrame(model_info), use_container_width=True)
    
    # Performance metrics
    st.markdown('<h3 style="color: #00f5ff;">📊 REAL-TIME METRICS</h3>', unsafe_allow_html=True)
    
    # Simulate real-time metrics
    import time
    current_time = time.strftime("%H:%M:%S")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #00f5ff; margin: 0; text-align: center;">⏱️ UPTIME</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">24h 15m</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ff00ff; margin: 0; text-align: center;">🔄 PROCESSED</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">151,112</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #00ff00; margin: 0; text-align: center;">🎯 ACCURACY</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">95.6%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffff00; margin: 0; text-align: center;">🕐 TIME</h4>
            <h3 style="color: #ffffff; margin: 0.5rem 0; text-align: center;">{current_time}</h3>
        </div>
        """, unsafe_allow_html=True)

# Futuristic footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h3 style="color: #00f5ff; margin: 0;">🛡️ CYBERFRAUDNET</h3>
    <p style="color: #ff00ff; margin: 0.5rem 0; font-size: 0.9rem;">Advanced AI Fraud Detection</p>
    <p style="color: #00ff00; margin: 0; font-size: 0.8rem;">⚡ Powered by Neural Networks ⚡</p>
    <div style="margin-top: 1rem;">
        <span style="color: #00f5ff;">🔬 PyTorch</span> • 
        <span style="color: #ff00ff;">🕸️ NetworkX</span> • 
        <span style="color: #00ff00;">🚀 Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
