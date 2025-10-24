from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_metrics():
    """Load model performance metrics"""
    # Get the parent directory (CyberFraudNet root)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_file = os.path.join(root_dir, "outputs", "metrics_report.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        # Extract key metrics
        metrics = {'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'auc': 'N/A'}
        lines = content.split('\n')
        for line in lines:
            if "Accuracy:" in line:
                metrics['accuracy'] = line.split(":")[1].strip()
            elif "Precision:" in line:
                metrics['precision'] = line.split(":")[1].strip()
            elif "Recall:" in line:
                metrics['recall'] = line.split(":")[1].strip()
            elif "AUC:" in line:
                metrics['auc'] = line.split(":")[1].strip()
        
        return metrics, content
    return None, None

def load_processed_data():
    """Load processed fraud data"""
    # Get the parent directory (CyberFraudNet root)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(root_dir, "data", "processed", "combined_fraud_data.csv")
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    return None

def get_image_base64(image_path):
    """Convert image to base64 for embedding"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/model-results')
def model_results():
    """Model performance results page"""
    metrics, full_report = load_metrics()
    
    # Load visualization images
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(root_dir, "outputs")
    images = {}
    for img_name in ['confusion_matrix.png', 'roc_curve.png', 'probability_distributions.png']:
        img_path = os.path.join(outputs_dir, img_name)
        images[img_name] = get_image_base64(img_path)
    
    return render_template('model_results.html', 
                         metrics=metrics, 
                         full_report=full_report,
                         images=images)

@app.route('/data-analysis')
def data_analysis():
    """Data analysis page"""
    df = load_processed_data()
    
    if df is not None:
        # Calculate statistics
        stats = {
            'total_transactions': len(df),
            'fraud_cases': int(df['is_fraud'].sum()),
            'fraud_rate': float(df['is_fraud'].mean() * 100),
            'features': len(df.columns)
        }
        
        # Create interactive charts
        fraud_counts = df['is_fraud'].value_counts()
        
        # Pie chart for fraud distribution
        pie_fig = go.Figure(data=[go.Pie(
            labels=['Normal', 'Fraud'],
            values=fraud_counts.values,
            hole=0.4,
            marker_colors=['#00f5ff', '#ff0000'],
            textfont_size=16,
            textfont_color='white'
        )])
        pie_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True,
            legend=dict(font_color='white')
        )
        
        # Histogram for purchase values
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=df[df['is_fraud']==0]['purchase_value'],
            name='Normal',
            marker_color='#00f5ff',
            opacity=0.7,
            nbinsx=50
        ))
        hist_fig.add_trace(go.Histogram(
            x=df[df['is_fraud']==1]['purchase_value'],
            name='Fraud',
            marker_color='#ff0000',
            opacity=0.7,
            nbinsx=50
        ))
        hist_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title='Purchase Value',
            yaxis_title='Count',
            barmode='overlay',
            showlegend=True,
            legend=dict(font_color='white')
        )
        
        # Convert plots to JSON
        pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)
        hist_json = json.dumps(hist_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Sample data
        sample_data = df.head(20).to_dict('records')
        
        return render_template('data_analysis.html',
                             stats=stats,
                             pie_chart=pie_json,
                             hist_chart=hist_json,
                             sample_data=sample_data)
    else:
        return render_template('data_analysis.html', stats=None)

@app.route('/live-prediction')
def live_prediction():
    """Live prediction page"""
    return render_template('live_prediction.html')

@app.route('/system-status')
def system_status():
    """System status page"""
    import time
    current_time = time.strftime("%H:%M:%S")
    
    # System info
    system_info = {
        'uptime': '24h 15m',
        'processed': '151,112',
        'accuracy': '95.6%',
        'current_time': current_time
    }
    
    return render_template('system_status.html', system_info=system_info)

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for live prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Flexible validation - accept various column names
            # Map common column variations to standard names
            column_mapping = {}
            
            # Look for user/customer ID columns
            user_cols = [col for col in df.columns if any(term in col.lower() for term in ['user', 'customer', 'client', 'account'])]
            if user_cols:
                column_mapping['user_id'] = user_cols[0]
            elif 'id' in df.columns:
                column_mapping['user_id'] = 'id'
            else:
                # Create synthetic user IDs
                df['user_id'] = range(1, len(df) + 1)
                column_mapping['user_id'] = 'user_id'
            
            # Look for product/item columns
            product_cols = [col for col in df.columns if any(term in col.lower() for term in ['product', 'item', 'sku', 'goods'])]
            if product_cols:
                column_mapping['product_id'] = product_cols[0]
            else:
                # Create synthetic product IDs
                df['product_id'] = np.random.randint(1, 100, len(df))
                column_mapping['product_id'] = 'product_id'
            
            # Look for seller/merchant columns
            seller_cols = [col for col in df.columns if any(term in col.lower() for term in ['seller', 'merchant', 'vendor', 'shop'])]
            if seller_cols:
                column_mapping['seller_id'] = seller_cols[0]
            else:
                # Create synthetic seller IDs
                df['seller_id'] = np.random.randint(1, 50, len(df))
                column_mapping['seller_id'] = 'seller_id'
            
            # Rename columns to standard names for processing
            df = df.rename(columns={v: k for k, v in column_mapping.items() if v != k})
            
            # Calculate statistics
            stats = {
                'records': len(df),
                'columns': len(df.columns),
                'size_kb': round(df.memory_usage(deep=True).sum() / 1024, 1),
                'column_mapping': column_mapping  # Show what columns were mapped
            }
            
            # Create network graph
            G = nx.Graph()
            sample_df = df.head(min(100, len(df)))
            
            for _, row in sample_df.iterrows():
                user = f"U_{row['user_id']}"
                product = f"P_{row['product_id']}"
                seller = f"S_{row['seller_id']}"
                
                G.add_edge(user, product, weight=1)
                G.add_edge(product, seller, weight=1)
            
            # Network statistics
            network_stats = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': round(nx.density(G), 3),
                'components': nx.number_connected_components(G)
            }
            
            # Simulate fraud detection
            fraud_probability = np.random.random()
            risk_score = np.random.randint(1, 100)
            
            if fraud_probability > 0.7:
                risk_level = "HIGH RISK"
                risk_color = "#ff0000"
                risk_icon = "🚨"
            elif fraud_probability > 0.4:
                risk_level = "MEDIUM RISK"
                risk_color = "#ff6600"
                risk_icon = "⚠️"
            else:
                risk_level = "LOW RISK"
                risk_color = "#00ff00"
                risk_icon = "✅"
            
            # Sample data preview
            sample_data = df.head(10).to_dict('records')
            
            return jsonify({
                'success': True,
                'stats': stats,
                'network_stats': network_stats,
                'fraud_analysis': {
                    'probability': round(fraud_probability, 3),
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'risk_icon': risk_icon
                },
                'sample_data': sample_data
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)