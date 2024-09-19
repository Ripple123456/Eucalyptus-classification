import os
from PIL import Image, UnidentifiedImageError

# Define the base directory where the 'train' and 'validation' folders are located
base_dir = '/Users/huangjinyi/Desktop/莉CV论文/图片备份'  # Replace with your actual base path

# List of subfolders for 'train' and 'validation'
subfolders = ['train', 'validation']

# Image augmentation functions
def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image(image):
    return image.rotate(180)

def mirror_image(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

# Apply augmentations and save the images in their respective folders
for subfolder in subfolders:
    for species_folder in os.listdir(os.path.join(base_dir, subfolder)):
        species_path = os.path.join(base_dir, subfolder, species_folder)
        if os.path.isdir(species_path):  # Check if it's a directory
            for image_name in os.listdir(species_path):
                if image_name.endswith('.jpg'):  # Check if the file is an image
                    image_path = os.path.join(species_path, image_name)
                    try:
                        with Image.open(image_path) as img:
                            # Flip the image
                            img_flipped = flip_image(img)
                            img_flipped.save(os.path.join(species_path, f'flipped_{image_name}'))
                            # Rotate the image
                            img_rotated = rotate_image(img)
                            img_rotated.save(os.path.join(species_path, f'rotated_{image_name}'))
                            # Mirror the image
                            img_mirrored = mirror_image(img)
                            img_mirrored.save(os.path.join(species_path, f'mirrored_{image_name}'))
                    except UnidentifiedImageError:
                        print(f"Cannot identify image file {image_path}, skipping.")
                    except Exception as e:
                        print(f"An error occurred with file {image_path}: {e}")

print("Image augmentation completed for train and validation sets.")
