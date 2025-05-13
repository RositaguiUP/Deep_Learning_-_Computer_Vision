import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import cv2

def resizeImages(original_folder, resized_folder, target_size=(128, 128)):
    for filename in os.listdir(original_folder):
        img = cv2.imread(os.path.join(original_folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(resized_folder, f"resized_{filename}"), img_resized)

def predictImages(input_folder, resized_folder, predicted_folder, version, target_size=(128, 128)):
    # Load the trained model
    model_save_path = f"models/encoder_decoder_model_{version}.h5" 
    model = load_model(model_save_path, custom_objects={'mse': MeanSquaredError()})
    model.summary()

    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            # cv2.imwrite(os.path.join(resized_folder, f"resized_{filename}"), img_resized)
            img = img_resized / 255.0  # Normalize to [0,1]
            img = np.expand_dims(img, axis=0) # Add batch dimension (1, H, W, C)

            predicted_image = model.predict(img)  # Output shape: (1, H, W, C)
            predicted_image = np.squeeze(predicted_image, axis=0)  # Remove batch dimension
            predicted_image = (predicted_image * 255).astype(np.uint8)  # Convert to uint8

            cv2.imwrite(os.path.join(predicted_folder, f"predicted_{version}_{filename}"), predicted_image)

input_folder = 'images/predictions/input'
resized_folder = 'images/predictions/resized'
predicted_folder = 'images/predictions/predicted'

version = "v5_aum5"

predictImages(input_folder, resized_folder, predicted_folder, version)

# output_folder = 'images/predictions/output'
# resized_out_folder = 'images/predictions/resized_output'
# resizeImages(input_folder, resized_folder)
# resizeImages(output_folder, resized_out_folder)

print("Done")