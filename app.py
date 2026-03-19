import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

# --- 1. MODEL LOADING ---
MODEL_PATH = 'model/skin_cancer_model.keras'
# We use compile=False to avoid issues with custom optimizers during loading
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASSES = [
    "Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis-like Lesions", 
    "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions"
]

DESC = {
    "Actinic Keratosis": "Precancerous lesion caused by sun damage. Should be treated early.",
    "Basal Cell Carcinoma": "Common skin cancer. Slow growing but requires surgical removal.",
    "Benign Keratosis-like Lesions": "Non-cancerous skin growth, often appearing waxy or scaly.",
    "Dermatofibroma": "Benign fibrous growth, typically firm and small.",
    "Melanoma": "High-risk malignancy. Requires immediate dermatological evaluation.",
    "Melanocytic Nevi": "Common benign mole. Monitor for the 'ABCDE' changes.",
    "Vascular Lesions": "Benign vascular tumors or malformations."
}

def get_risk_data(label):
    if label in ["Melanoma", "Basal Cell Carcinoma"]:
        return "High", "#ff4d4d"
    if label == "Actinic Keratosis":
        return "Moderate", "#ffa500"
    return "Low", "#4ade80"

# --- 2. GRAD-CAM LOGIC ---
def generate_gradcam(img_array, model):
    try:
        pre_layers = []
        base_model = None
        post_layers = []
        
        found_base = False
        for layer in model.layers:
            if hasattr(layer, 'layers') and layer.name.startswith('efficientnet'):
                base_model = layer
                found_base = True
                continue
                
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
                
            if not found_base:
                pre_layers.append(layer)
            else:
                post_layers.append(layer)

        if base_model is None:
            return None
            
        x = img_array
        for layer in pre_layers:
            x = layer(x, training=False)
            
        with tf.GradientTape() as tape:
            conv_outputs = base_model(x, training=False)
            tape.watch(conv_outputs)
            
            preds = conv_outputs
            for layer in post_layers:
                preds = layer(preds, training=False)
            
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)[0]
        gate_f = tf.reduce_mean(grads, axis=(0, 1))
        
        heatmap = conv_outputs[0] @ gate_f[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess for EfficientNet
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    
    # CRITICAL: Use the official EfficientNet preprocessing
    # This prevents the "same class every time" error
    x = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediction
    preds = model.predict(x)[0]
    pred_idx = np.argmax(preds)
    label = CLASSES[pred_idx]
    
    risk_lvl, risk_clr = get_risk_data(label)

    # Format all probabilities for the UI bars
    all_probs = []
    for i, p in enumerate(preds):
        all_probs.append({
            "name": CLASSES[i], 
            "probability": round(float(p) * 100, 2)
        })
    # Sort by highest probability first
    all_probs = sorted(all_probs, key=lambda x: x['probability'], reverse=True)

    # Generate Heatmap
    heatmap = generate_gradcam(x, model)
    heatmap_base64 = ""
    
    if heatmap is not None:
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        _, buffer = cv2.imencode('.png', superimposed)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    # Determine if ABCDE warning is needed
    abcde_relevant = True if label in ["Melanocytic Nevi", "Melanoma"] else False

    return jsonify({
        "prediction": label,
        "confidence": round(float(preds[pred_idx]) * 100, 2),
        "description": DESC.get(label, "No description available."),
        "risk_level": risk_lvl,
        "risk_color": risk_clr,
        "all_probs": all_probs,
        "heatmap": heatmap_base64,
        "abcde_relevant": abcde_relevant
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)