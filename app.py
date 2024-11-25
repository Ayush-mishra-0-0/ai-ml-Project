import flask
from flask import Flask, request, jsonify, render_template
import torch
import pickle
import cv2
import numpy as np
import pandas as pd
from utils.train import OptimizedCLIPModel, Config
from sklearn.preprocessing import OneHotEncoder
import logging
from flask import Response
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return flask.json.JSONEncoder.default(self)

# Set custom JSON encoder
app.json_encoder = NumpyEncoder

# Load models and configurations
def load_models(clip_path='models/best_model.pt', catboost_path='models/catboost_model.sav'):
    try:
        # Load CLIP model
        CFG = Config('cuda' if torch.cuda.is_available() else 'cpu')
        clip_model = OptimizedCLIPModel(CFG)
        checkpoint = torch.load(clip_path, map_location=CFG.device)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
        clip_model.eval()
        clip_model = clip_model.to(CFG.device)

        # Load CatBoost model
        catboost_model = pickle.load(open(catboost_path, 'rb'))
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError(f"Error loading models: {e}")

    return clip_model, catboost_model, CFG

def load_xgboost_models():
    try:
        # Load XGBoost models and OneHotEncoder
        xgb_max_temp = pickle.load(open('models/xgb_max_temp_model.pkl', 'rb'))
        xgb_min_temp = pickle.load(open('models/xgb_min_temp_model.pkl', 'rb'))
        encoder = pickle.load(open('models/condition_encoder.pkl', 'rb'))
    except Exception as e:
        logger.error(f"Error loading XGBoost models: {e}")
        raise RuntimeError(f"Error loading XGBoost models: {e}")

    return xgb_max_temp, xgb_min_temp, encoder

# Image preprocessing
def process_image(image_file):
    # Read image from file object
    image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if len(image_np.shape) == 2:  # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    image_np = cv2.resize(image_np, (244, 244))
    image_np = np.moveaxis(image_np, -1, 0)  # Move channels first
    return image_np

# Predict cloud cover
def predict_cloudcover(image, clip_model, catboost_model, cfg):
    processed_image = process_image(image)
    image_tensor = torch.tensor(np.stack([processed_image]), device=cfg.device, dtype=torch.float32)

    with torch.no_grad():
        features = clip_model.image_encoder(image_tensor)
        embeddings = clip_model.image_projection(features)
    
    prediction = catboost_model.predict(features.cpu().numpy())
    return float(prediction[0])
import numpy as np

def predict_temperature(cloud_cover, humidity, condition, xgb_max_temp, xgb_min_temp, encoder):
    # Define input values
    input_data = {
        "Cloud Cover (%)": cloud_cover,   # Example value
        "Humidity (%)": humidity,      # Example value
        "Condition": condition     # Example condition
    }
    input_df = pd.DataFrame([input_data])

    # One-hot encode the 'Condition' column using the saved encoder
    encoded_conditions = encoder.transform(input_df[['Condition']])
    encoded_columns = encoder.get_feature_names_out(input_features=['Condition'])
    
    # Convert the ndarray to a list to ensure it can be serialized
    encoded_conditions_list = encoded_conditions.tolist()
    
    # Convert encoded data to DataFrame for compatibility
    encoded_df = pd.DataFrame(encoded_conditions_list, columns=encoded_columns, index=input_df.index)
    
    # Concatenate encoded features with other features
    prepared_input = pd.concat([input_df[['Cloud Cover (%)', 'Humidity (%)']], encoded_df], axis=1)

    # Ensure all columns match the trained model
    # Fill missing columns with 0 (if any) to match training data
    all_columns = list(xgb_max_temp.feature_names_in_)
    prepared_input = prepared_input.reindex(columns=all_columns, fill_value=0)

    # Make predictions
    max_temp = xgb_max_temp.predict(prepared_input)
    min_temp = xgb_min_temp.predict(prepared_input)

    return max_temp, min_temp


# Load models globally
clip_model, catboost_model, cfg = load_models()
xgb_max_temp, xgb_min_temp, encoder = load_xgboost_models()

# Routes
@app.route('/')
def index():
    """Render the main index page"""
    return render_template('index.html')

@app.route('/predict/cloudcover', methods=['POST'])
def cloud_cover_prediction():
    """Endpoint for cloud cover prediction"""
    try:
        # Check if image is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        
        # Perform cloud cover prediction
        cloud_cover = predict_cloudcover(image_file, clip_model, catboost_model, cfg)
        
        return jsonify({'cloud_cover': cloud_cover})
    
    except Exception as e:
        logger.error(f"Cloud cover prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/temperature', methods=['POST'])
def temperature_prediction():
    """Endpoint for temperature prediction"""
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['cloud_cover', 'humidity', 'condition']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Perform temperature prediction
        max_temp, min_temp = predict_temperature(
            float(data['cloud_cover']), 
            float(data['humidity']), 
            data['condition'], 
            xgb_max_temp, 
            xgb_min_temp, 
            encoder
        )
        return Response(
            json.dumps({'max_temp': max_temp[0], 'min_temp': min_temp[0]}, cls=NumpyEncoder),
            mimetype='application/json'
        )
        return jsonify({'max_temp': max_temp[0], 'min_temp': min_temp[0]}, cls=NumpyEncoder)
    
    except Exception as e:
        logger.error(f"Temperature prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Error Handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad Request'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource Not Found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)