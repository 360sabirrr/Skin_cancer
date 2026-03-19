import os
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = r"C:\Users\sabir\OneDrive\Desktop\Skin_cancer_detection_system"
val_dir = os.path.join(base_dir, "Data", "val")
model_path = os.path.join(base_dir, "model", "skin_cancer_model.keras")

print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

# We need to compile to evaluate loss and accuracy
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Preparing validation dataset...")
target_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMG_SIZE = 224 # Assuming B0 was used
BATCH_SIZE = 32

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=target_classes,
    shuffle=False
)

print("Evaluating...")
loss, accuracy = model.evaluate(val_ds, verbose=1)

print("\n--- RESULTS ---")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")
