# Skin Cancer Detection System

This is a deep learning-based Skin Cancer Detection System that uses an EfficientNet model to classify skin lesions into 7 different categories. It provides a web interface (built with Flask) to upload an image and receive a predicted diagnosis, confidence score, risk assessment, and a Grad-CAM heatmap highlighting the areas of the lesion the AI focused on.

## Features
- **7-Class Classification:** Detects Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis-like Lesions, Dermatofibroma, Melanoma, Melanocytic Nevi, and Vascular Lesions.
- **Grad-CAM Visualization:** Generates a heatmap overlaid on the original image to provide visual interpretability of the AI's decision.
- **Risk Level Assessment:** Categorizes the predicted skin lesion into Low, Moderate, or High risk.
- **Web Interface:** A sleek, dark-themed UI built with TailwindCSS for easy image uploading and result viewing.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/360sabirrr/Skin_cancer.git
   cd Skin_cancer
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```


## Project Structure
- `app.py`: The main Flask application handling routing, model inference, and Grad-CAM generation.
- `train.py`: The script used to train the EfficientNet model using TensorFlow/Keras.
- `prepare_data.py`: A utility script for organizing the HAM10000 dataset into train/val splits.
- `model/skin_cancer_model.keras`: The pre-trained deep learning model.
- `templates/index.html`: The frontend user interface.
- `requirements.txt`: Python package dependencies.

## Disclaimer
This tool is for educational and informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
