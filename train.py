import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generators (IMPORTANT: grayscale)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_datagen.flow_from_directory(
    'train',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'   # ✅ FIXED
)

# Load testing data
test_data = test_datagen.flow_from_directory(
    'test',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'   # ✅ FIXED
)

# Check classes
print("Classes:", train_data.class_indices)

# Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(7, activation='softmax')   # ✅ FIXED (7 classes)
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# Save model
model.save("emotion_model.h5")

print("✅ Model trained and saved successfully!")

