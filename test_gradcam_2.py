import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('model/skin_cancer_model.keras', compile=False)

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
        print(f"Error: {e}")
        return None

img = np.random.rand(1, 224, 224, 3).astype(np.float32)
heatmap = generate_gradcam(img, model)

if heatmap is not None:
    print("SUCCESS! Heatmap shape:", heatmap.shape)
else:
    print("FAILED")
