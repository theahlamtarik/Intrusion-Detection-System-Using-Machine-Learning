import os
import logging
import warnings
from flask import Flask, render_template, request, flash, redirect, url_for, session
from werkzeug.exceptions import HTTPException  
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import sklearn

# Configure warnings and logging
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ML Models')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_PATH = os.path.join(DATASET_DIR, 'kddcup.csv')

# Model configurations
MODELS = {
    'random_forest': 'random_forest_model.pkl',
    'decision_tree': 'decision_tree_model.pkl',
    'logistic_regression': 'logistic_regression_model.pkl'
}

ENCODERS = {
    'protocol': 'protocol_type_label_encoder.pkl',
    'service': 'service_label_encoder.pkl',
    'flag': 'flag_label_encoder.pkl'
}

def load_all_models_and_encoders():
    loaded_models = {}
    loaded_encoders = {}
    try:
        for name, filename in MODELS.items():
            path = os.path.join(MODEL_DIR, filename)
            with open(path, 'rb') as file:
                loaded_models[name] = pickle.load(file)
                logger.info(f"Loaded {name} model")
        
        for name, filename in ENCODERS.items():
            path = os.path.join(MODEL_DIR, filename)
            with open(path, 'rb') as file:
                loaded_encoders[name] = pickle.load(file)
                logger.info(f"Loaded {name} encoder")
        return loaded_models, loaded_encoders
    except Exception as e:
        logger.error(f"Error loading models/encoders: {str(e)}")
        raise

# Initialize Flask and load models
app = Flask(__name__)
app.secret_key = os.urandom(24)

try:
    models, encoders = load_all_models_and_encoders()
    model = models['random_forest']  # Set default model
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded successfully: {len(df)} rows")
    
except Exception as e:
    logger.error(f"Startup error: {str(e)}")
    raise

print("Intrusion Detection System initialized successfully!")
print(f"Dataset loaded: {DATASET_PATH}")
print("IDS is running without any errors!")

fields = [
    {
      "format": "default",
      "name": "duration",
      "type": "number",
      "description": "duration of connection in seconds"
    },
    {
      "format": "default",
      "name": "protocol_type",
      "type": "string",
      "description": "connection protocol (tcp, udp, icmp)"
    },
    {
      "format": "default",
      "name": "service",
      "type": "string",
      "description": "dst port mapped to service (E.G: http, ftp,..)"
    },
    {
      "format": "default",
      "name": "flag",
      "type": "string",
      "description": "normal or error status flag of connection"
    },
    {
      "format": "default",
      "name": "src_bytes",
      "type": "number",
      "description": "number of databytes from src to dst"
    },
    {
      "format": "default",
      "name": "dst_bytes",
      "type": "any",
      "description": "bytes from dst to src"
    },
    {
      "format": "default",
      "name": "land",
      "type": "number",
      "description": "1 if connection is from/to the same host/port; else 0"
    },
    {
      "format": "default",
      "name": "wrong_fragment",
      "type": "number",
      "description": "number of 'wrong' fragments (values 0,1,3)"
    },
    {
      "format": "default",
      "name": "urgent",
      "type": "number",
      "description": "number of urgent packets"
    },
    {
      "format": "default",
      "name": "hot",
      "type": "number",
      "description": "number of hot indicators"
    },
    {
      "format": "default",
      "name": "num_failed_logins",
      "type": "number",
      "description": "number of failed login attempts"
    },
    {
      "format": "default",
      "name": "logged_in",
      "type": "number",
      "description": "1 if successfully logged in; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lnum_compromised",
      "type": "number",
      "description": "number of compromised conditions"
    },
    {
      "format": "default",
      "name": "lroot_shell",
      "type": "number",
      "description": "1 if root shell is obtained; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lsu_attempted",
      "type": "number",
      "description": "1 if su root command attempted; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lnum_root",
      "type": "number",
      "description": "number of root accesses"
    },
    {
      "format": "default",
      "name": "lnum_file_creations",
      "type": "number",
      "description": "number of file creation operations"
    },
    {
      "format": "default",
      "name": "lnum_shells",
      "type": "number",
      "description": "number of shell prompts "
    },
    {
      "format": "default",
      "name": "lnum_access_files",
      "type": "number",
      "description": "number of operations on access control files"
    },
    {
      "format": "default",
      "name": "lnum_outbound_cmds",
      "type": "number",
      "description": "number of outbound commands in an ftp session"
    },
    {
      "format": "default",
      "name": "is_host_login",
      "type": "number",
      "description": "1 if the login belongs to the hot list; 0 otherwise "
    },
    {
      "format": "default",
      "name": "is_guest_login",
      "type": "number",
      "description": "1 if the login is a guest login; 0 otherwise"
    },
    {
      "format": "default",
      "name": "count",
      "type": "number",
      "description": "number of connections to the same host as the current connection in the past two seconds"
    },
    {
      "format": "default",
      "name": "srv_count",
      "type": "number",
      "description": "number of connections to the same service as the current connection in the past two seconds"
    },
    {
      "format": "default",
      "name": "serror_rate",
      "type": "number",
      "description": "% of connections that have SYN errors"
    },
    {
      "format": "default",
      "name": "srv_serror_rate",
      "type": "number",
      "description": "% of connections that have SYN errors "
    },
    {
      "format": "default",
      "name": "rerror_rate",
      "type": "number",
      "description": "% of connections that have REJ errors"
    },
    {
      "format": "default",
      "name": "srv_rerror_rate",
      "type": "number",
      "description": "% of connections that have REJ errors"
    },
    {
      "format": "default",
      "name": "same_srv_rate",
      "type": "number",
      "description": "% of connections to the same service"
    },
    {
      "format": "default",
      "name": "diff_srv_rate",
      "type": "number",
      "description": "% of connections to different services"
    },
    {
      "format": "default",
      "name": "srv_diff_host_rate",
      "type": "number",
      "description": "% of connections to different hosts"
    },
    {
      "format": "default",
      "name": "dst_host_count",
      "type": "number",
      "description": "count of connections having same dst host"
    },
    {
      "format": "default",
      "name": "dst_host_srv_count",
      "type": "number",
      "description": "count of connections having same dst host and using same service"
    },
    {
      "format": "default",
      "name": "dst_host_same_srv_rate",
      "type": "number",
      "description": "% of connections having same dst port and using same service"
    },
    {
      "format": "default",
      "name": "dst_host_diff_srv_rate",
      "type": "number",
      "description": "% of different services on current host"
    },
    {
      "format": "default",
      "name": "dst_host_same_src_port_rate",
      "type": "number",
      "description": "% of connections to current host having same src port"
    },
    {
      "format": "default",
      "name": "dst_host_srv_diff_host_rate",
      "type": "number",
      "description": "% of connections to same service coming from different hosts"
    },
    {
      "format": "default",
      "name": "dst_host_serror_rate",
      "type": "number",
      "description": "% of connections to current host that have S0 error"
    },
    {
      "format": "default",
      "name": "dst_host_srv_serror_rate",
      "type": "number",
      "description": "% of connections to current host and specified service that have an S0 error"
    },
    {
      "format": "default",
      "name": "dst_host_rerror_rate",
      "type": "number",
      "description": "% of connections to current host that have an RST error"
    },
    {
      "format": "default",
      "name": "dst_host_srv_rerror_rate",
      "type": "number",
      "description": "% of connections to the current host and specified service that have an RST error"
    },
    {
      "format": "default",
      "name": "label",
      "type": "string",
      "description": "specifies whether normal traffic or attack in the network"
    }
]

@app.before_request
def before_request():
    # Log all requests
    logger.info(f"Request: {request.method} {request.path}")

@app.route("/")
def index():
    return render_template('index.html', title="IDS Home")

@app.route("/features")
def features():
    try :
        df = pd.read_csv('dataset/kddcup.csv')
        if df.empty:
            raise ValueError("Dataset is empty")
            
        df_head = df.head(4)
        table_html = df_head.to_html(
            classes='table table-striped table-hover', 
            index=False,
            border=0
        )
        return render_template('features.html', 
                             table_html=table_html,
                             fields=fields,
                             title="Network Features")
                             
    except FileNotFoundError:
        logger.error("Dataset file not found: dataset/kddcup.csv")
        return render_template('error.html',
                             error_title="Dataset Not Found",
                             error_message="The required dataset file is missing.")
    except Exception as e:
        logger.error(f"Error in features route: {str(e)}")
        return render_template('error.html',
                             error_title="System Error",
                             error_message="An unexpected error occurred.")

@app.route("/pda")
def pda():
  return render_template('pda.html')

@app.route("/results", methods=['POST'])
def results():
    if request.method == 'POST':
        try:
            logger.info("Form data received: %s", request.form)
            input_mode = request.form.get('inputMode', 'manual')
            logger.info("Input mode: %s", input_mode)

            # Basic required fields regardless of mode
            required_fields = [
                'duration', 'protocolType', 'service', 'flag',
                'srcBytes', 'dstnBytes', 'wrongFragment',
                'loggedIn', 'samePortCount', 'sameDstnCount'
            ]
            
            # Validate required fields
            for field in required_fields:
                if not request.form.get(field):
                    raise ValueError(f"Field cannot be empty: {field}")

            # Process form data
            try:
                input_data = {
                    'duration': float(request.form['duration']),
                    'protocol_type': request.form['protocolType'].lower(),
                    'service': request.form['service'].lower(),
                    'flag': request.form['flag'].upper(),
                    'src_bytes': float(request.form['srcBytes']),
                    'dst_bytes': float(request.form['dstnBytes']),
                    'wrong_fragment': float(request.form['wrongFragment']),
                    'logged_in': float(request.form['loggedIn']),
                    'srv_count': float(request.form['samePortCount']),
                    'dst_host_count': float(request.form['sameDstnCount'])
                }
            except (ValueError, KeyError) as e:
                raise ValueError(f"Invalid input format: {str(e)}")

            # Log the processed input data
            logger.info("Processed input data: %s", input_data)

            # Load models
            dt = pickle.load(open('ML Models/decision_tree_model.pkl', 'rb'))
            lr = pickle.load(open('ML Models/logistic_regression_model.pkl', 'rb'))
            rf = pickle.load(open('ML Models/random_forest_model.pkl', 'rb'))

            # Load and apply encoders
            encoders = {}
            for feature in ['protocol_type', 'service', 'flag']:
                with open(f'ML Models/{feature}_label_encoder.pkl', 'rb') as f:
                    encoders[feature] = pickle.load(f)
                try:
                    input_data[feature] = encoders[feature].transform([input_data[feature]])[0]
                except ValueError as e:
                    raise ValueError(f"Invalid value for {feature}: {input_data[feature]}")

            # Prepare input data for prediction
            data = [[
                input_data['duration'],
                input_data['protocol_type'],
                input_data['service'],
                input_data['flag'],
                input_data['src_bytes'],
                input_data['dst_bytes'],
                input_data['wrong_fragment'],
                input_data['logged_in'],
                input_data['srv_count'],
                input_data['dst_host_count']
            ]]

            # Make predictions
            predictions = {
                'dt': dt.predict(data)[0],
                'rf': rf.predict(data)[0],
                'lr': lr.predict(data)[0]
            }

            # Log predictions
            logger.info("Model predictions - DT: %s, RF: %s, LR: %s", 
                       predictions['dt'], predictions['rf'], predictions['lr'])

            # Calculate accuracies
            accuracies = {}
            if input_mode == 'predefined' and request.form.get('attackType'):
                y_true = request.form['attackType'].lower()
                accuracies = {
                    'dt': f"{accuracy_score([y_true], [predictions['dt']]):.2%}",
                    'rf': f"{accuracy_score([y_true], [predictions['rf']]):.2%}",
                    'lr': f"{accuracy_score([y_true], [predictions['lr']]):.2%}"
                }
            else:
                accuracies = {
                    'dt': "N/A (Manual Entry)",
                    'rf': "N/A (Manual Entry)",
                    'lr': "N/A (Manual Entry)"
                }

            # Log accuracies
            logger.info("Accuracies: %s", accuracies)

            return render_template('results.html',
                                dt_prediction=predictions['dt'],
                                rf_prediction=predictions['rf'],
                                lr_prediction=predictions['lr'],
                                dt_accuracy=accuracies['dt'],
                                rf_accuracy=accuracies['rf'],
                                lr_accuracy=accuracies['lr'])

        except Exception as e:
            logger.error("Error processing request: %s", str(e), exc_info=True)
            return render_template('error.html',
                                error_title="Processing Error",
                                error_message=str(e))

@app.errorhandler(500)
def handle_500(e):
    logger.error(f"500 error: {str(e)}")
    return render_template('error.html',
                         error_title="Server Error",
                         error_message="An internal server error occurred."), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return render_template('error.html',
                         error_title="Error",
                         error_message=str(e)), 500

@app.errorhandler(HTTPException)
def handle_exception(e):
    logger.error(f"HTTP error occurred: {str(e)}")
    return render_template('error.html',
                         error_title=f"Error {e.code}",
                         error_message=e.description), e.code

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled error: {str(e)}")
    return render_template('error.html',
                         error_title="Server Error",
                         error_message="An unexpected error occurred"), 500

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for development