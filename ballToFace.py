import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import matplotlib.pyplot as plt


def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize to [0,1]
            images.append(img)
    print(f"loaded {len(images)} images from {folder}")
    return np.array(images)

# Data Augmentation function
def augment_image(image, label):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    # Apply the SAME augmentation to both image & label
    augmented = data_augmentation(tf.concat([image, label], axis=-1))
    return augmented[..., :3], augmented[..., 3:]  # Split into input-output

# Load dataset with augmentation applied
def load_dataset(face_folder, sphere_folder, augment=True, num_augmented=5):
    face_images = load_images_from_folder(face_folder)
    sphere_images = load_images_from_folder(sphere_folder)

    augmented_faces, augmented_spheres = [], []
    
    if augment:
        for img, label in zip(face_images, sphere_images):
            for _ in range(num_augmented):  # Create multiple augmented versions
                aug_img, aug_label = augment_image(img, label)
                augmented_faces.append(aug_img)
                augmented_spheres.append(aug_label)

    # Convert to NumPy arrays
    augmented_faces = np.array(augmented_faces)
    augmented_spheres = np.array(augmented_spheres)

    # Combine original & augmented images
    full_faces = np.concatenate([face_images, augmented_faces]) if augment else face_images
    full_spheres = np.concatenate([sphere_images, augmented_spheres]) if augment else sphere_images

    return train_test_split(full_faces, full_spheres, test_size=0.2, random_state=42)


# def load_dataset(face_folder, sphere_folder):
#     face_images_original = load_images_from_folder(face_folder)
#     face_images_augmented = augment_dataset(face_images_original, num_augmented=2)
#     face_images_all = np.concatenate([face_images_original, face_images_augmented])
#     sphere_images = load_images_from_folder(sphere_folder)
#     return train_test_split(face_images_all, sphere_images, test_size=0.2, random_state=42)


def build_encoder_decoder(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(8 * 8 * 512, activation='relu')(x)
    x = layers.Reshape((8, 8, 512))(x)
    
    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    model = models.Model(inputs, x)
    return model

# Create model
model = build_encoder_decoder()
model.compile(optimizer='adam', loss='mse')
model.summary()

# Load dataset
face_folder = 'images/input'
sphere_folder = 'images/output'
train_images, val_images, train_labels, val_labels = load_dataset(face_folder, sphere_folder, augment=True)

# Training setup
def train_model(model, train_images, train_labels, val_images, val_labels, epochs=50, batch_size=32):
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )
    return model, history

# Train the model
trained_model, history = train_model(model, train_images, train_labels, val_images, val_labels)

# Save the trained model
model_save_path = "encoder_decoder_model_v5_aum5.h5"
trained_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

test_loss = trained_model.evaluate(val_images, val_labels)
print(f"Validation Loss: {test_loss:.4f}")




train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plot the loss curves
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')  # 'bo-' = blue dots + line
plt.plot(epochs, val_loss, 'r^-', label='Validation Loss')  # 'r^-' = red triangles + line
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()
