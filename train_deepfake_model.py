"""
DEEPFAKE MODEL TRAINING - VS CODE LOCAL VERSION
================================================
Train the model locally using your NPZ dataset.
Designed to work perfectly with the backend.

Usage:
    python train_deepfake_model.py

This will:
1. Load NPZ dataset from Google Drive or local
2. Train model with proper architecture
3. Save .h5 to backend models directory
4. Model immediately ready for inference
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import gc

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # INPUT: Path to your NPZ file (change this to your path)
    'dataset_path': r'C:\Projects\DF-Analysis\backend\models\detection\loaded_dataset.npz',
    
    # OUTPUT: Where to save the trained model
    'model_save_path': r'C:\Projects\DF-Analysis\backend\models\detection\deepfake_model_trained.h5',
    
    # Model parameters (MUST match backend expectations)
    'frames_per_video': 30,
    'image_size': 96,
    'batch_size': 4,
    'epochs': 30,
    'learning_rate': 5e-5,
}

# ============================================
# STEP 1: LOAD NPZ DATASET
# ============================================
print("\n" + "="*70)
print("STEP 1: LOADING NPZ DATASET")
print("="*70)

if not os.path.exists(CONFIG['dataset_path']):
    print(f"\n❌ ERROR: Dataset not found at {CONFIG['dataset_path']}")
    print("Please update 'dataset_path' in CONFIG to point to your NPZ file")
    sys.exit(1)

print(f"Loading from: {CONFIG['dataset_path']}")

try:
    data = np.load(CONFIG['dataset_path'], allow_pickle=True)
    X_full = data['X']
    y_full = data['y']
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: X={X_full.shape}, y={y_full.shape}")
    print(f"  Real samples: {np.sum(y_full==0)}")
    print(f"  Fake samples: {np.sum(y_full==1)}")
    print(f"  Value range: [{X_full.min():.3f}, {X_full.max():.3f}]")
    
except Exception as e:
    print(f"❌ ERROR loading NPZ: {e}")
    sys.exit(1)

# Normalize if needed
if X_full.max() > 1.0:
    X_full = X_full / 255.0
    print(f"  Normalized to [0, 1]")

# ============================================
# STEP 2: DATA AUGMENTATION
# ============================================
print("\n" + "="*70)
print("STEP 2: DATA AUGMENTATION")
print("="*70)

print("Applying augmentations...")
X_flip = np.flip(X_full, axis=3)  # Horizontal flip
X_vflip = np.flip(X_full, axis=2)  # Vertical flip

X_aug = np.concatenate([X_full, X_flip, X_vflip], axis=0)
y_aug = np.concatenate([y_full, y_full, y_full], axis=0)

print(f"✓ After augmentation: {X_aug.shape}")

# Shuffle
indices = np.random.permutation(len(X_aug))
X_aug = X_aug[indices]
y_aug = y_aug[indices]

del X_full, y_full
gc.collect()

# ============================================
# STEP 3: TRAIN/VAL/TEST SPLIT
# ============================================
print("\n" + "="*70)
print("STEP 3: TRAIN/VAL/TEST SPLIT")
print("="*70)

# 70% train+val, 30% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)

# 80% train, 20% val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

print(f"✓ Train: {X_train.shape}")
print(f"✓ Val: {X_val.shape}")
print(f"✓ Test: {X_test.shape}")

del X_aug, y_aug, X_train_val, y_train_val
gc.collect()

# ============================================
# STEP 4: BUILD MODEL
# ============================================
print("\n" + "="*70)
print("STEP 4: BUILDING MODEL ARCHITECTURE")
print("="*70)

print("\nBuilding CNN feature extractor (MobileNetV2)...")
frame_input = Input(shape=(CONFIG['image_size'], CONFIG['image_size'], 3), name='frame_input')
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_tensor=frame_input)

# Freeze all layers for feature extraction
for layer in mobilenet.layers:
    layer.trainable = False

cnn_pool = GlobalAveragePooling2D()(mobilenet.output)
cnn_model = Model(inputs=frame_input, outputs=cnn_pool, name='cnn_feature_extractor')

print(f"✓ CNN model: {cnn_model.input_shape} → {cnn_model.output_shape}")

print("\nBuilding full video model with TimeDistributed...")
video_input = Input(shape=(CONFIG['frames_per_video'], CONFIG['image_size'], CONFIG['image_size'], 3), name='video_input')

# TimeDistributed CNN
x = TimeDistributed(cnn_model, name='td_features')(video_input)

# Temporal LSTM layers
x = LSTM(128, return_sequences=True, name='lstm_1')(x)
x = Dropout(0.3)(x)
x = LSTM(64, return_sequences=False, name='lstm_2')(x)
x = BatchNormalization()(x)

# Dense classification layers
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=video_input, outputs=output, name='deepfake_detector')

print(f"✓ Full model built")
print(f"  Input: {model.input_shape}")
print(f"  Output: {model.output_shape}")

# Compile
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model compiled")

# Show architecture
print("\nModel Architecture:")
print("-" * 70)
model.summary()

# ============================================
# STEP 5: TRAINING
# ============================================
print("\n" + "="*70)
print("STEP 5: TRAINING MODEL")
print("="*70)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        CONFIG['model_save_path'],
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining on {len(X_train)} samples...")
print(f"Validating on {len(X_val)} samples...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 6: EVALUATION
# ============================================
print("\n" + "="*70)
print("STEP 6: EVALUATION")
print("="*70)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✓ Test Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc*100:.2f}%")

# ============================================
# STEP 7: SAVE MODEL
# ============================================
print("\n" + "="*70)
print("STEP 7: SAVING MODEL")
print("="*70)

os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)

model.save(CONFIG['model_save_path'])

print(f"✓ Model saved to: {CONFIG['model_save_path']}")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# ============================================
# STEP 8: VERIFICATION
# ============================================
print("\n" + "="*70)
print("STEP 8: VERIFICATION - TESTING BACKEND LOADING")
print("="*70)

try:
    # Try to load the model to verify it's valid
    loaded_model = tf.keras.models.load_model(CONFIG['model_save_path'])
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  Input: {loaded_model.input_shape}")
    print(f"  Output: {loaded_model.output_shape}")
    
    # Test prediction
    test_input = np.random.randn(1, 30, 96, 96, 3).astype(np.float32) / 255.0
    pred = loaded_model.predict(test_input, verbose=0)
    
    print(f"\n✓ Test prediction: {pred[0][0]:.4f}")
    print(f"\n✅ MODEL IS READY FOR BACKEND!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("Model loading failed - something went wrong!")
    sys.exit(1)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel saved at: {CONFIG['model_save_path']}")
print("Backend will automatically load this model.")
