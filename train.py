import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#  1. CONFIGURATION
base_dir = r"C:\Users\sabir\OneDrive\Desktop\Skin_cancer_detection_system"
train_dir = os.path.join(base_dir, "Data", "train")
val_dir = os.path.join(base_dir, "Data", "val")

# SETTINGS: Toggle between B0 (Faster/CPU) and B4 (Accurate/GPU)
# For B0: IMG_SIZE=224, BATCH_SIZE=32
# For B4: IMG_SIZE=380, BATCH_SIZE=16 (or 8 if memory is low)
USE_MODEL = "B0" 
IMG_SIZE = 224 if USE_MODEL == "B0" else 380
BATCH_SIZE = 32 if USE_MODEL == "B0" else 16
EPOCHS = 10

# HARDCODED CLASSES: This prevents the "8th class/val folder" error
target_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(target_classes)

# 2. LOAD DATASETS 
print(f"🚀 Loading Datasets for {USE_MODEL}...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=target_classes  # Strictly enforces 7 classes
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=target_classes
)

# Optimize performance for CPU/GPU
autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

# 3. DATA AUGMENTATION 
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

#4. BUILD MODEL 
print(f"Building EfficientNet{USE_MODEL} Model...")

if USE_MODEL == "B0":
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
else:
    base_model = EfficientNetB4(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False  # Freeze base layers for Phase 1

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. CALLBACKS
model_folder = os.path.join(base_dir, 'model')
os.makedirs(model_folder, exist_ok=True)

callbacks = [
    ModelCheckpoint(os.path.join(model_folder, 'skin_cancer_model.keras'), save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

#6. TRAINING 
print("\n--- Phase 1: Training Classification Head ---")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

print("\n--- Phase 2: Fine-Tuning Full Model ---")
base_model.trainable = True  # Unfreeze everything
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n Training Complete! Model saved as 'skin_cancer_model.keras' in {model_folder}")