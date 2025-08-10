import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ✅ 1. Load Dataset
batch_size = 4  # Reduced for small dataset
img_size = (128, 128)

dataset = tf.keras.utils.image_dataset_from_directory(
    "cats_and_dogs",           # path to your dataset folder
    labels="inferred",
    label_mode="binary",       # for binary classification
    image_size=img_size,
    batch_size=batch_size
)

# ✅ 2. Normalize and Augment Images
normalization_layer = tf.keras.layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

dataset = dataset.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))

# ✅ 3. Shuffle and Prefetch
dataset = dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# ✅ 4. Check Dataset Size and Split
dataset_size = dataset.cardinality().numpy()
print(f"Total batches: {dataset_size}")

if dataset_size < 2:
    raise ValueError("🚫 Dataset too small. Add more images or reduce batch size.")

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_ds = dataset.take(train_size).repeat()
val_ds = dataset.skip(train_size).repeat()

# ✅ 5. Build CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output: Cat or Dog
])

# ✅ 6. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ 7. Train the Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=train_size,
    validation_steps=val_size
)

# ✅ 8. Evaluate Model
loss, acc = model.evaluate(val_ds, steps=val_size)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")

# ✅ 9. Predict New Image
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print("🐶 It's a Dog!" if prediction[0][0] > 0.5 else "🐱 It's a Cat!")

# 🔍 Example usage:
predict_image("cats_and_dogs/Cats/cat10.jpg")