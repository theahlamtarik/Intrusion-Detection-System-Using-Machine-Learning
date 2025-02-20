# Network Intrusion Detection System (IDS)

A machine learning-based web application for detecting network intrusions using multiple ML algorithms. Built with Flask and scikit-learn.

## Features

- **Multiple ML Models**
  - Random Forest
  - Decision Tree
  - Logistic Regression
- **Real-time Analysis**
  - Manual input mode
  - Predefined scenarios
- **Comprehensive Visualization**
  - Preliminary Data Analysis
  - Network Feature Descriptions
  - Model Accuracy Comparisons

## Technology Stack

- Python 3.x
- Flask
- scikit-learn
- pandas
- Bootstrap 5
- Pickle (for model serialization)

## Project Structure

```
IDS_ML_FLASK/
├── ML Models/              # Trained ML models and encoders
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── logistic_regression_model.pkl
├── dataset/               # Dataset directory
│   └── kddcup.csv        # Network traffic dataset
├── static/               # Static files
│   ├── css/
│   └── images/
├── templates/            # HTML templates
├── start.py             # Main application file
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/theahlamtarik/Intrusion-Detection-System-Using-Machine-Learning.git
cd IDS_ML_FLASK
```

2. Download the dataset
- Go to [NSL-KDD Dataset on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Download `KDDTest+.csv` and `KDDTrain+.csv`
- Rename the combined file to `kddcup.csv`
- Place it in the `dataset` folder

3. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run the application
```bash
python start.py
```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Choose input mode:
   - Manual: Enter network parameters manually
   - Predefined: Select from pre-configured attack scenarios
3. Submit for analysis
4. View results from multiple ML models with accuracy scores

## Features Description

The system analyzes various network parameters including:
- Protocol type
- Service
- Connection duration
- Source/destination bytes
- Login attempts
- Error rates
- And more...



