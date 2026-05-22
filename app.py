import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from ml_pipeline import MLPipeline
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize ML Pipeline
ml_pipeline = MLPipeline()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/eda')
def eda():
    try:
        # Check if dataset is loaded
        if not ml_pipeline.is_data_loaded():
            flash('Please upload a dataset first.', 'warning')
            return redirect(url_for('home'))
        
        # Generate EDA plots
        plot_path = ml_pipeline.generate_eda_plots()
        
        # Get basic dataset info
        dataset_info = ml_pipeline.get_dataset_info()
        
        return render_template('eda.html', 
                             plot_path=plot_path.replace('static/', ''),
                             dataset_info=dataset_info)
    except Exception as e:
        flash(f'Error generating EDA: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/performance')
def performance():
    try:
        # Check if dataset is loaded
        if not ml_pipeline.is_data_loaded():
            flash('Please upload a dataset first.', 'warning')
            return redirect(url_for('home'))
        
        # Train all models and get performance metrics
        performance_results = ml_pipeline.train_all_models()
        
        return render_template('performance.html', 
                             results=performance_results)
    except Exception as e:
        flash(f'Error training models: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/prediction')
def prediction():
    # Get feature names for the form
    feature_names = ml_pipeline.get_feature_names()
    return render_template('prediction.html', feature_names=feature_names)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'dataset' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('home'))
    
    file = request.files['dataset']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('home'))
    
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load dataset into ML pipeline
            ml_pipeline.load_dataset(filepath)
            flash('Dataset uploaded and loaded successfully!', 'success')
        except Exception as e:
            flash(f'Error loading dataset: {str(e)}', 'danger')
    else:
        flash('Invalid file format. Please upload a CSV file.', 'danger')
    
    return redirect(url_for('home'))

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get form data
        form_data = {}
        feature_names = ml_pipeline.get_feature_names()
        
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                flash(f'Please provide a value for {feature}', 'danger')
                return redirect(url_for('prediction'))
            try:
                form_data[feature] = float(value)
            except ValueError:
                # Handle categorical features
                form_data[feature] = value
        
        # Make prediction
        predictions = ml_pipeline.predict_single(form_data)
        
        flash('Prediction completed successfully!', 'success')
        return render_template('prediction.html', 
                             feature_names=feature_names,
                             predictions=predictions,
                             input_data=form_data)
    
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'danger')
        return redirect(url_for('prediction'))

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'batch_file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('prediction'))
    
    file = request.files['batch_file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('prediction'))
    
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make batch predictions
            results_path = ml_pipeline.predict_batch(filepath)
            flash('Batch prediction completed successfully!', 'success')
            return send_file(results_path, as_attachment=True, 
                           download_name='batch_predictions.csv')
        except Exception as e:
            flash(f'Error making batch prediction: {str(e)}', 'danger')
    else:
        flash('Invalid file format. Please upload a CSV file.', 'danger')
    
    return redirect(url_for('prediction'))

@app.route('/download_results/<filename>')
def download_results(filename):
    try:
        filepath = os.path.join(RESULTS_FOLDER, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
