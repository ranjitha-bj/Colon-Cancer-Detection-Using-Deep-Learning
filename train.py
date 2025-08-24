from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Image size and paths
img_size = 224
train_path = "dataset/train"
val_path = "dataset/val"

# Create 'model/' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Data generators with normalization
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, zoom_range=0.2, rotation_range=10)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Load images
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary'
)

# Print class mapping
print("Class indices:", train_gen.class_indices)

# Load base model without top
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

# Final model
model = Model(inputs=base_model.input, outputs=preds)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("model/colon_model.h5", save_best_only=True, verbose=1)

# Train model
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[checkpoint])

# Evaluate
loss, acc = model.evaluate(val_gen)
print(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")
