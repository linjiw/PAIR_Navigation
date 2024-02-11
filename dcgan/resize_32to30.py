import os
import numpy as np
from PIL import Image

def process_image(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to 30x30
    img = img.resize((30, 30), Image.Resampling.LANCZOS)  # Updated resizing method

    # Convert to numpy array
    img_array = np.array(img)

    # Apply threshold
    threshold = 127
    img_array = np.where(img_array > threshold, 0, 1)

    return img_array
def print_image_as_dots(image_array):
    for row in image_array:
        for pixel in row:
            # Print a filled dot for 1, space for 0
            print('●' if pixel == 1 else ' ', end='')
        print()  # Newline after each row
def process_folder(folder_path):
    processed_images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Assuming the images are in PNG format
            image_path = os.path.join(folder_path, filename)
            processed_img = process_image(image_path)
            processed_images.append(processed_img)

    return processed_images

# Example usage
folder_path = 'sampled_images/'
processed_images = process_folder(folder_path)

# Now processed_images is a list of 30x30 numpy arrays
for img_array in processed_images:
    print(f"Image shape: {img_array.shape}")
    print_image_as_dots(img_array)
    print("\n" + "="*30 + "\n")  # Separator between images