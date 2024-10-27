import torch
import cv2
import numpy as np
import gradio as gr
import pickle
from train import Config, OptimizedCLIPModel
import os

def load_models(clip_path='models/best_model.pt', catboost_path='models/catboost_model.sav'):
    # Load CLIP model
    CFG = Config('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = OptimizedCLIPModel(CFG)
    checkpoint = torch.load(clip_path, map_location=CFG.device)
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    clip_model.eval()
    clip_model = clip_model.to(CFG.device)
    
    # Load CatBoost model
    catboost_model = pickle.load(open(catboost_path, 'rb'))
    
    return clip_model, catboost_model, CFG

def process_image(image):
    # Convert to RGB if needed and resize
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, (244, 244))
    # Move channels first
    image = np.moveaxis(image, -1, 0)
    return image

def predict_cloudcover(image, clip_model, catboost_model, cfg):
    # Process image
    processed_image = process_image(image)
    
    # Convert to tensor
    image_tensor = torch.tensor(np.stack([processed_image]), 
                              device=cfg.device, 
                              dtype=torch.float32)
    
    # Get CLIP features
    with torch.no_grad():
        features = clip_model.image_encoder(image_tensor)
        embeddings = clip_model.image_projection(features)
    
    # Get CatBoost prediction
    prediction = catboost_model.predict(features.cpu().numpy())
    
    return float(prediction[0])

# Load models
clip_model, catboost_model, cfg = load_models()

# Create Gradio interface
def predict_image(input_image):
    prediction = predict_cloudcover(input_image, clip_model, catboost_model, cfg)
    return f"Predicted Cloud Cover: {prediction:.2f}%"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Prediction"),
    title="Sky Image Cloud Cover Predictor",
    description="Upload a sky image to predict the percentage of cloud cover.",
    examples=[
        "dataset/data_images/Extracted Images/20160101084000.jpg",  
        "dataset/data_images/Extracted Images/20160101095000.jpg"
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)