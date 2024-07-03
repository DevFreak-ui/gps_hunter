import os
import shutil
import random

# Paths to the original dataset folders
dataset_path = 'dataset'
jpeg_images_path = os.path.join(dataset_path, 'JPEGImages')
annotations_path = os.path.join(dataset_path, 'Annotations')

# Paths to the new dataset folders
dataset_2_path = 'dataset_2'
images_path = os.path.join(dataset_2_path, 'images')
annotations_path_2 = os.path.join(dataset_2_path, 'annotations')

# Create the new dataset folder structure
os.makedirs(images_path, exist_ok=True)
os.makedirs(annotations_path_2, exist_ok=True)

# Get a list of all image files
image_files = [f for f in os.listdir(jpeg_images_path) if f.endswith('.jpg')]

# Shuffle the image files to ensure random selection
random.shuffle(image_files)

# List to store selected files
selected_files = []

# Select 10 images that have corresponding annotation files
for image_file in image_files:
    # Construct the corresponding annotation file name
    annotation_file = image_file.replace('.jpg', '.xml')
    src_annotation_path = os.path.join(annotations_path, annotation_file)
    
    # Check if the annotation file exists
    if os.path.exists(src_annotation_path):
        selected_files.append((image_file, annotation_file))
        if len(selected_files) == 10:
            break

# Copy the selected images and their corresponding annotation files
for image_file, annotation_file in selected_files:
    # Construct the full paths to the source files
    src_image_path = os.path.join(jpeg_images_path, image_file)
    src_annotation_path = os.path.join(annotations_path, annotation_file)
    
    # Copy the image file to the new images folder
    shutil.copy(src_image_path, images_path)
    
    # Copy the annotation file to the new annotations folder
    shutil.copy(src_annotation_path, annotations_path_2)
    
    print(f'Image file: {image_file} --> Matching annotation file: {annotation_file}')

print('Selected 10 images and their annotations have been copied to dataset_2.')
