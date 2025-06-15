import os
import pandas as pd
import numpy as np
import joblib
import json
import plotly
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data and model
train_df = None
model = None
encoders = {}
features = []
numerical_cols = []
categorical_cols = []
target_column = 'churned'

# Load model and encoders if they exist
model = None
encoders = {}

def load_model():
    global model, encoders, features, numerical_cols, categorical_cols, target_column
    
    # Reset globals
    model = None
    encoders = {}
    features = []
    numerical_cols = []
    categorical_cols = []
    
    # Try to load the model with metadata
    try:
        model_data = joblib.load('customer_churn_model.pkl')
        
        if isinstance(model_data, dict):
            # New format with metadata
            if 'model' in model_data:
                model = model_data['model']
                features = model_data.get('features', [])
                numerical_cols = model_data.get('numerical_cols', [])
                categorical_cols = model_data.get('categorical_cols', [])
                target_column = model_data.get('target_column', 'Churn')
                print(f"Loaded model with {len(features)} features and {len(encoders)} encoders")
            else:
                print("Warning: Model dictionary missing 'model' key")
        else:
            # Legacy format - just the model
            model = model_data
            print("Warning: Loaded legacy model (no metadata). Feature matching may not work correctly.")
            
            # Try to infer features if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
                print(f"Inferred {len(features)} features from model")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
    
    # Load encoders if they exist
    if os.path.exists('encoders.pkl'):
        try:
            encoders = joblib.load('encoders.pkl')
            if not isinstance(encoders, dict):
                print("Warning: encoders.pkl did not contain a dictionary")
                encoders = {}
            else:
                print(f"Loaded {len(encoders)} categorical encoders")
                # Update categorical columns from encoders if not set
                if not categorical_cols and encoders:
                    categorical_cols = list(encoders.keys())
                    print(f"Inferred {len(categorical_cols)} categorical columns from encoders")
        except Exception as e:
            print(f"Error loading encoders: {str(e)}")
    else:
        print("No encoders.pkl found - categorical encoding will not be available")
        encoders = {}

# Load model on startup
load_model()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def load_data(file_path):
    """Load and preprocess the dataset"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Basic preprocessing
    df = df.drop_duplicates()
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Handle missing values without chained assignment
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def get_data_info(df):
    """Get basic information about the dataset"""
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        'churn_rate': round(df[target_column].mean() * 100, 2) if target_column in df.columns else 0
    }
    return info

def get_numerical_analysis(df, column):
    """Generate analysis for numerical columns"""
    stats = df[column].describe().to_dict()
    
    # Create histogram
    fig = px.histogram(df, x=column, title=f'Distribution of {column}', 
                      color_discrete_sequence=['#3498db'])
    
    # Create box plot for outliers
    fig_box = px.box(df, y=column, points=False, 
                    title=f'Box Plot of {column} (Outliers Detection)',
                    color_discrete_sequence=['#e74c3c'])
    
    return {
        'stats': stats,
        'chart': json.loads(fig.to_json()),
        'box_plot': json.loads(fig_box.to_json())
    }

def get_categorical_analysis(df, column):
    """Generate analysis for categorical columns"""
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['Value', 'Count']
    
    # Limit to top 20 categories if too many
    if len(value_counts) > 20:
        value_counts = value_counts.head(20)
    
    # Create bar chart
    fig = px.bar(value_counts, x='Value', y='Count', 
                title=f'Distribution of {column}',
                color_discrete_sequence=['#2ecc71'])
    
    # Create pie chart for top categories
    fig_pie = px.pie(value_counts, values='Count', names='Value',
                    title=f'Percentage of {column} (Top 10)',
                    color_discrete_sequence=px.colors.sequential.RdBu)
    
    return {
        'value_counts': value_counts.to_dict('records'),
        'chart': json.loads(fig.to_json()),
        'pie_chart': json.loads(fig_pie.to_json())
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/upload', methods=['GET', 'POST'])
def upload_train():
    global train_df, numerical_cols, categorical_cols, target_column
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Update target column if provided
            if 'target' in request.form and request.form['target']:
                target_column = request.form['target']
            
            # Load and preprocess data
            train_df = load_data(filepath)
            
            # Identify numerical and categorical columns
            numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target column from features
            if target_column in numerical_cols:
                numerical_cols.remove(target_column)
            elif target_column in categorical_cols:
                categorical_cols.remove(target_column)
            
            # Get data info for display
            data_info = get_data_info(train_df)
            
            # Generate data preview
            preview = train_df.head(10).to_html(classes='table table-striped table-bordered', index=False)
            
            return render_template('upload_train.html', 
                                 data_preview=preview,
                                 num_rows=data_info['rows'],
                                 num_cols=data_info['columns'],
                                 missing_values=data_info['missing_values'],
                                 active_page='upload_train')
    
    return render_template('upload_train.html', active_page='upload_train')

@app.route('/analyze')
def analyze():
    global train_df, numerical_cols, categorical_cols
    
    if train_df is None:
        flash('Please upload training data first', 'warning')
        return redirect(url_for('upload_train'))
    
    # Get data info
    data_info = get_data_info(train_df)
    
    # Get data types
    data_types = train_df.dtypes.reset_index()
    data_types.columns = ['Column', 'Data Type']
    data_types_html = data_types.to_html(classes='table table-striped table-bordered', index=False)
    
    # Generate basic insights
    insights = []
    if data_info['missing_values'] > 0:
        insights.append(f"⚠️ Your dataset has {data_info['missing_values']}% missing values that might need to be handled.")
    else:
        insights.append("✅ No missing values detected in your dataset.")
    
    if data_info['churn_rate'] < 5 or data_info['churn_rate'] > 70:
        insights.append(f"⚠️ Your dataset has an imbalanced target variable ({data_info['churn_rate']}% churn rate). Consider using techniques like SMOTE or class weights.")
    else:
        insights.append(f"✅ Your dataset has a balanced target variable ({data_info['churn_rate']}% churn rate).")
    
    return render_template('analyze.html', 
                         num_rows=data_info['rows'],
                         num_cols=data_info['columns'],
                         missing_values=data_info['missing_values'],
                         churn_rate=data_info['churn_rate'],
                         data_types=data_types_html,
                         numerical_cols=numerical_cols,
                         categorical_cols=categorical_cols,
                         insights=insights,
                         active_page='analyze')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, encoders, train_df
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess the test data
            test_df = load_data(filepath)
            
            # Make predictions
            if model and encoders:
                # Check if model has predict_proba method
                if not hasattr(model, 'predict_proba'):
                    flash('Model does not support probability predictions', 'danger')
                    return redirect(url_for('predict'))
                
                # Check if we have features to work with
                if not features:
                    flash('No features available for prediction. Please train a model first.', 'danger')
                    return redirect(url_for('train_model'))
                
                # Check if required features are available in the test data
                available_features = [f for f in features if f in test_df.columns]
                missing_features = list(set(features) - set(available_features))
                
                if missing_features:
                    flash(f'Warning: Missing {len(missing_features)} features. Using available features only.', 'warning')
                
                if not available_features:
                    flash('No matching features found between training and test data', 'danger')
                    return redirect(url_for('predict'))
                
                try:
                    # Log the features being used
                    print(f"Using features: {available_features}")
                    
                    # Ensure we only use the features that exist in both training and test data
                    X_test = test_df[available_features].copy()
                    
                    # Apply any necessary preprocessing (like encoding)
                    for col in categorical_cols:
                        if col in X_test.columns and col in encoders:
                            # Convert to string and handle unknown categories
                            X_test[col] = X_test[col].astype(str).apply(
                                lambda x: x if x in encoders[col].classes_ else 'Unknown'
                            )
                            # Transform using the fitted encoder
                            X_test[col] = encoders[col].transform(X_test[col])
                    
                    # Ensure we have the same column order as during training
                    X_test = X_test[available_features]
                    
                    # Convert to numpy array for prediction
                    X_test_array = X_test.values
                    
                    # Make predictions
                    print(f"Making prediction on array of shape: {X_test_array.shape}")
                    if X_test_array.shape[1] == 0:
                        raise ValueError("No valid features available for prediction")
                        
                    predictions = model.predict_proba(X_test_array)[:, 1]
                except Exception as e:
                    flash(f'Error making predictions: {str(e)}', 'danger')
                    return redirect(url_for('predict'))
                test_df['churn_probability'] = predictions
                test_df['predicted_churn'] = (predictions >= float(request.form.get('threshold', 0.5))).astype(int)
                
                # Generate prediction summary
                churn_count = test_df['predicted_churn'].sum()
                non_churn_count = len(test_df) - churn_count
                churn_rate = (churn_count / len(test_df)) * 100
                
                # Get top 10 risky customers
                top_risky = test_df.nlargest(10, 'churn_probability')
                top_risky_html = top_risky[['customerID', 'churn_probability']].to_html(
                    classes='table table-striped table-bordered', 
                    index=False,
                    float_format='{:.2f}'.format
                )
                
                # Get feature importance if available
                feature_importance = []
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance = [
                        {'feature': feature, 'importance': float(importance[i])} 
                        for i, feature in enumerate(features)
                    ]
                    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
                
                # Prepare prediction results
                prediction_results = {
                    'churn_count': churn_count,
                    'non_churn_count': non_churn_count,
                    'churn_rate': round(churn_rate, 2),
                    'top_risky_customers': top_risky_html,
                    'feature_importance': feature_importance,
                    'probability_distribution': predictions.tolist()
                }
                
                # Save predictions to CSV
                output_file = os.path.join(app.config['UPLOAD_FOLDER'], f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                test_df.to_csv(output_file, index=False)
                
                return render_template('predict.html', 
                                     predictions=prediction_results,
                                     active_page='predict')
            else:
                flash('Model not trained yet. Please train the model first.', 'danger')
                return redirect(url_for('train_model'))
    
    return render_template('predict.html', active_page='predict')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    global model, train_df, features, numerical_cols, categorical_cols, target_column, encoders
    
    if train_df is None:
        flash('Please upload training data first', 'warning')
        return redirect(url_for('upload_train'))
    
    if request.method == 'POST':
        # Get model parameters from form
        model_type = request.form.get('model_type', 'random_forest')
        
        try:
            # Identify feature columns if not already set
            if not features:
                # Exclude non-feature columns
                non_feature_cols = ['customerID', 'Churn', 'churned', 'churn', 'Churned']
                features = [col for col in train_df.columns if col not in non_feature_cols]
                
                # Identify numerical and categorical columns
                numerical_cols = train_df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = list(set(features) - set(numerical_cols))
            
            # Prepare features and target
            X = train_df[features].copy()
            y = train_df[target_column].copy()
            
            # Encode categorical variables
            from sklearn.preprocessing import LabelEncoder
            
            # Reset encoders
            encoders = {}
            
            # Encode target variable
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Encode categorical features
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
            
            # Save encoders
            joblib.dump(encoders, 'encoders.pkl')
            
            # Train model based on selected type
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            else:  # gradient_boosting
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            # Train the model
            model.fit(X, y)
            
            # Prepare model data with metadata
            model_data = {
                'model': model,
                'features': features.copy(),  # Store a copy to prevent reference issues
                'numerical_cols': numerical_cols.copy(),
                'categorical_cols': categorical_cols.copy(),
                'target_column': target_column,
                'model_type': model_type,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_names_in_': features  # For scikit-learn compatibility
            }
            
            # Save the model and metadata
            joblib.dump(model_data, 'customer_churn_model.pkl')
            
            # Also save the model's feature names for scikit-learn's get_feature_names_out
            if hasattr(model, 'feature_names_in_'):
                model.feature_names_in_ = features
            
            # Save encoders for categorical variables
            if encoders:
                joblib.dump(encoders, 'encoders.pkl')
                print(f"Saved {len(encoders)} encoders to encoders.pkl")
            
            print(f"Model saved with {len(features)} features")
            print(f"Numerical features: {len(numerical_cols)}")
            print(f"Categorical features: {len(categorical_cols)}")
            print(f"Target column: {target_column}")
            
            flash(f'Model trained successfully with {model_type.replace("_", " ").title()}!', 'success')
            return redirect(url_for('analyze'))
            
        except Exception as e:
            flash(f'Error training model: {str(e)}', 'danger')
    
    return render_template('train.html', active_page='train')

@app.route('/download_predictions')
def download_predictions():
    # Get the most recent predictions file
    import glob
    list_of_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_*.csv'))
    
    if not list_of_files:
        flash('No prediction files found', 'danger')
        return redirect(url_for('predict'))
    
    latest_file = max(list_of_files, key=os.path.getctime)
    return send_file(latest_file, as_attachment=True)

# API Endpoints for AJAX requests
@app.route('/api/analyze/numerical/<column>')
def api_numerical_analysis(column):
    global train_df
    if train_df is None or column not in train_df.columns:
        return jsonify({'error': 'Invalid column or no data loaded'}), 400
    
    analysis = get_numerical_analysis(train_df, column)
    return jsonify(analysis)

@app.route('/api/analyze/categorical/<column>')
def api_categorical_analysis(column):
    global train_df
    if train_df is None or column not in train_df.columns:
        return jsonify({'error': 'Invalid column or no data loaded'}), 400
    
    analysis = get_categorical_analysis(train_df, column)
    return jsonify(analysis)

@app.route('/api/analyze/correlations')
def api_correlations():
    global train_df, numerical_cols
    if train_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Calculate correlations for numerical columns
    corr = train_df[numerical_cols + [target_column]].corr()
    
    # Create heatmap
    fig = px.imshow(corr, 
                   labels=dict(x="Features", y="Features", color="Correlation"),
                   x=corr.columns,
                   y=corr.columns,
                   color_continuous_scale='RdBu_r',
                   zmin=-1, zmax=1)
    
    # Get top correlations with target
    target_corr = corr[target_column].drop(target_column).sort_values(ascending=False)
    top_correlations = [
        {'feature': feature, 'correlation': float(corr)} 
        for feature, corr in target_corr.items()
    ]
    
    return jsonify({
        'heatmap': json.loads(fig.to_json()),
        'top_correlations': top_correlations
    })

@app.route('/download_sample')
def download_sample():
    """Route to download the sample dataset"""
    try:
        # Check if sample file exists, if not create one
        sample_file = 'sample_customer_churn.csv'
        if not os.path.exists(sample_file):
            # Create a sample dataset if it doesn't exist
            data = {
                'customerID': ['0001-AAAA', '0002-BBBB', '0003-CCCC', '0004-DDDD', '0005-EEEE'],
                'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
                'SeniorCitizen': [0, 1, 0, 0, 1],
                'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
                'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
                'tenure': [1, 34, 5, 45, 2],
                'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
                'MultipleLines': ['No', 'Yes', 'No', 'No phone service', 'No'],
                'InternetService': ['DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic'],
                'OnlineSecurity': ['No', 'No', 'No', 'Yes', 'No'],
                'OnlineBackup': ['No', 'Yes', 'No', 'No', 'No'],
                'DeviceProtection': ['No', 'No', 'Yes', 'Yes', 'No'],
                'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
                'StreamingTV': ['No', 'No', 'No', 'No', 'No'],
                'StreamingMovies': ['No', 'No', 'No', 'No', 'No'],
                'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
                'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
                'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check'],
                'MonthlyCharges': [29.85, 56.95, 42.30, 60.50, 75.00],
                'TotalCharges': [29.85, 1889.50, 211.50, 2800.00, 150.00],
                'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
            }
            sample_df = pd.DataFrame(data)
            sample_df.to_csv(sample_file, index=False)
        
        return send_file(
            sample_file,
            as_attachment=True,
            download_name='sample_customer_churn.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        flash(f'Error downloading sample file: {str(e)}', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
