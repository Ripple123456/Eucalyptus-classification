import os
import shutil
from sklearn.model_selection import train_test_split

# Define the base directory where the data folders are located
base_dir = '/Users/huangjinyi/Desktop/莉CV论文/图片备份'



# List of the folders containing images
folders = [
    'Angophora hispida',
    'Eucalyptus cinerea',
    'Eucalyptus deglupta',
    'Eucalyptus globulus',
    'Eucalyptus Pauciflora'
]

# Create train and validation directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# Split data into training and validation sets (80:20 ratio) and copy files
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        continue
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    train_files, validation_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Create subdirectories for training and validation sets
    train_subdir = os.path.join(train_dir, folder)
    validation_subdir = os.path.join(validation_dir, folder)
    
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)
    if not os.path.exists(validation_subdir):
        os.makedirs(validation_subdir)
    
    # Copy images to the corresponding subdirectories
    for f in train_files:
        dst_file = os.path.join(train_subdir, os.path.basename(f))
        if not os.path.exists(dst_file):  # Check if the file already exists in the destination
            shutil.copy2(f, dst_file)
    for f in validation_files:
        dst_file = os.path.join(validation_subdir, os.path.basename(f))
        if not os.path.exists(dst_file):  # Check if the file already exists in the destination
            shutil.copy2(f, dst_file)

print("Data has been split into training and validation sets.")
