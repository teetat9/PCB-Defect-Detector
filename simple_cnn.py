import tensorflow as tf
from tensorflow import keras
from data_loader import PCBDataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load Data
loader = PCBDataLoader("organized_dataset")
X_train, X_test, y_train, y_test = loader.split_data()

# Normalize (Normally RGB ranges from 0 - 255 -> 0 - 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build model
model = keras.Sequential([
    # First conv block
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    # Second conv block
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    # Third conv block
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    # Dense layer
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(6, activation='softmax') # 6 defect types -> result is probability of each defect
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# Train with callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1
                    )

# Evaluate
val_loss, val_acc = model.evaluate(X_test, y_test)
print(f"\n{'='*50}")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"{'='*50}")

# Save model
model.save('pcb_defect_model.h5')
print("\nModel saved as 'pcb_defect_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved as 'training_history.png'")
plt.show()