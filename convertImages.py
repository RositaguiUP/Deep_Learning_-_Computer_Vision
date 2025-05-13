import os
from PIL import Image

# Function to convert CR2 to JPG
def convert_cr2_to_jpg(cr2_path, jpg_path):
    with Image.open(cr2_path) as img:
        img.save(jpg_path, 'JPEG')  # Save as JPG format

# Function to process all CR2 files in the given folder and convert to JPG
def convert_all_cr2_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.CR2'):
            cr2_path = os.path.join(input_folder, filename)
            jpg_filename = f"{os.path.splitext(filename)[0]}.jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            convert_cr2_to_jpg(cr2_path, jpg_path)
            print(f"Converted {filename} to {jpg_filename}")

# Paths to your input and output folders
input_orig_folder = 'images/input_original_02'
output_orig_folder = 'images/output_original_02'
input_folder = 'images/input'
output_folder = 'images/output'

# Convert all CR2 files in the input folder
convert_all_cr2_to_jpg(input_orig_folder, input_folder)

# Convert all CR2 files in the output folder
convert_all_cr2_to_jpg(output_orig_folder, output_folder)