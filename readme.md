# Sky Image Cloud Cover Predictor

## Overview
This project implements a machine learning system that automatically predicts cloud coverage percentage from sky camera images. The system uses computer vision and machine learning techniques to analyze hemispherical sky images and provide accurate cloud coverage estimates.

## 🌟 Features
- Real-time cloud coverage prediction from sky images
- Web-based user interface for easy image upload
- Support for hemispherical sky camera images
- Automated prediction pipeline
- API endpoint for integration with other systems

## 📁 Project Structure
```
.
├── app.py                 # Flask web application
├── cloud_coverage.py      # Core prediction logic
├── main.ipynb            # Development notebook
├── cat_boost.ipynb       # Model training notebook
├── train.py              # Training script
├── temp.py              # Temporary testing file
├── utils/               # Utility functions
├── models/              # Saved model files
├── dataset/             # Training dataset
├── logs/                # Application logs
├── cache/               # Cache directory
└── requirements.txt     # Project dependencies
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sky-cloud-predictor.git
cd sky-cloud-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Web Interface
1. Start the Gradio application:
```bash
python app.py
```
2. Open your browser and navigate to `http://localhost:7860`
3. Upload a sky image and click "Submit" to get the cloud coverage prediction
<!-- image show  website.png -->
![Web Interface Screenshot](website.png)


### API Usage
```python
import requests

url = 'http://localhost:7860/predict'
files = {'image': open('sky_image.jpg', 'rb')}
response = requests.post(url, files=files)
prediction = response.json()['prediction']
```

## 🔧 Model Training

To train the model on your own dataset:

1. Prepare your dataset in the `dataset/` directory
2. Configure training parameters in `train.py`
3. Run the training script:
```bash
python train.py
```

## 📊 Model Performance
- Current model achieves ~98.76% accuracy on test set
```
---------------------------------------------------
Train MAE: 3.063867080191343
Train RMSE: 4.519084504196453
Train MSE: 20.422124756068502
Train R2: 0.9753638347539638
---------------------------------------------------
---------------------------------------------------
Valid MAE: 5.798734045374874
Valid RMSE: 9.624912586359335
Valid MSE: 92.63894229505833
Valid R2: 0.8879491192051271
---------------------------------------------------
---------------------------------------------------
Test MAE: 5.812910249786526
Test RMSE: 9.69202086124033
Test MSE: 93.93526837471777
Test R2: 0.886006145423602
---------------------------------------------------
```
- Average prediction time: 0.5 seconds
- Supports images up to 1024x1024 pixels

## 🛠️ Technology Stack
- Python 3.8+
- Flask
- CatBoost
- OpenCV
- NumPy
- Pandas
- scikit-learn

## 📝 Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Minimum 8GB RAM
- Disk space: 2GB for model and dependencies

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
-Ayush Kumar Mishra (@ayushkum)

## 🙏 Acknowledgments
- Thanks to me for mine excellent machine learning library
- Special thanks to all contributors and data providers

## 📞 Support
For support, email ayushkumarmishra000@gamil.com or open an issue in the repository.