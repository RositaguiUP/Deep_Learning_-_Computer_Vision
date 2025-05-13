import os

folder = 'images/input_original_02'  # Change this to your folder path
files = os.listdir(folder)

# Filter files if needed (e.g., to only rename .jpg files)
files = [f for f in files if f.endswith('.CR2')]

for i, file in enumerate(files, start=101):
    old_name = os.path.join(folder, file)
    new_name = os.path.join(folder, f"image_{i:03d}.CR2")
    os.rename(old_name, new_name)

print("Renaming complete!")