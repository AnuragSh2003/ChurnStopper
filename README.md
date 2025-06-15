# Customer Churn Prediction System

A web-based application for analyzing customer data and predicting churn using machine learning.

## Features

- **Data Upload**: Upload your training and test datasets in CSV format
- **Exploratory Data Analysis**: Visualize and understand your data with interactive charts
- **Model Training**: Train machine learning models with different algorithms
- **Predictions**: Make predictions on new data and download results
- **Visualizations**: Interactive visualizations for better insights

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/churn-prediction-system.git
   cd churn-prediction-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Follow these steps**:
   - Upload your training dataset (CSV format)
   - Explore the data using the analysis tools
   - Train a machine learning model
   - Upload test data to make predictions
   - Download the prediction results

## Project Structure

```
churn-prediction-system/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── uploads/              # Directory for uploaded files
├── static/               # Static files (CSS, JS, images)
│   └── css/
│       └── style.css
└── templates/            # HTML templates
    ├── base.html         # Base template
    ├── index.html        # Home page
    ├── upload_train.html # Data upload page
    ├── analyze.html      # Data analysis page
    ├── train.html        # Model training page
    └── predict.html      # Prediction page
```

## Data Format

Your dataset should be in CSV format with the following requirements:

- Include a unique customer ID column (e.g., 'customerID')
- Include a target column indicating churn (e.g., 'churned' with values 0/1 or Yes/No)
- Ensure all columns have appropriate data types
- Handle or remove any sensitive information before uploading

## Supported Models

- **Random Forest**: Best for most datasets, handles non-linear relationships well
- **Gradient Boosting**: Often provides higher accuracy but may overfit with small datasets
- **Logistic Regression**: Fast and works well with linearly separable data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For any questions or issues, please open an issue on the GitHub repository.
