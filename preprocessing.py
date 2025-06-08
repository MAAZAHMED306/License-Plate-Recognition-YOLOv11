from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

#################################################################
### Split the Data
#################################################################

root_dir = "C:/Users/MAAZ/Source-Code/datasets/car-number-plate/"  # Absolute root directory for the dataset
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

def file_paths(root, valid_formats):
    "Get the full path to each image/label in the dataset"
    file_paths = []

    # Loop over the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Loop over the filenames in the current directory
        for filename in filenames:
            # Extract the file extension from the filename
            extension = os.path.splitext(filename)[1].lower()

            # If the filename has a valid extension, we build the full path to the file and append it to our list
            if extension in valid_formats:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

    return file_paths


# Absolute paths for images and labels
image_paths = file_paths("C:/Users/MAAZ/Source-Code/datasets/car-number-plate/images", valid_formats[:3])
label_paths = file_paths("C:/Users/MAAZ/Source-Code/datasets/car-number-plate/labels", valid_formats[-1])

# Split the data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)

def write_to_file(images_path, labels_path, X):
    # Create the directories if they don't exist
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Loop over the image paths
    for img_path in X:
        # Get the image name and extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1]
        
        # Read the image (ensure it's in the correct format)
        image = cv2.imread(img_path)
        
        # Save the image to the images directory
        cv2.imwrite(f"{images_path}/{img_name}{img_ext}", image)

        # Try to match the label file by stripping the prefix
        label_file_path = None
        for label_path in label_paths:
            # Matching based on image name (using startswith or equality check to account for naming differences)
            if os.path.splitext(os.path.basename(label_path))[0] == img_name:
                label_file_path = label_path
                break

        if label_file_path is None:
            print(f"Warning: Label file for {img_name} not found!")
            continue

        # Write the label to the new label file
        with open(f"{labels_path}/{img_name}.txt", "w") as f:
            with open(label_file_path, "r") as label_file:
                f.write(label_file.read())

# Write the train, validation, and test sets
write_to_file("C:/Users/MAAZ/Source-Code/datasets/images/train", "C:/Users/MAAZ/Source-Code/datasets/labels/train", X_train)
write_to_file("C:/Users/MAAZ/Source-Code/datasets/images/valid", "C:/Users/MAAZ/Source-Code/datasets/labels/valid", X_val)
write_to_file("C:/Users/MAAZ/Source-Code/datasets/images/test", "C:/Users/MAAZ/Source-Code/datasets/labels/test", X_test)

#################################################################
### Create a YAML file
#################################################################

# Create a dictionary with the paths to the train, valid, and test sets
data = {
    "path": "../datasets",  # Dataset root dir (can also use the full path to the `datasets` folder)
    "train": "images/train",  # Train images (relative to 'path')
    "val": "images/valid",    # Validation images (relative to 'path')
    "test": "images/test",    # Test images (optional)

    # Classes
    "names": ["number plate"]
}

# Write the dictionary to a YAML file
with open("number-plate.yaml", "w") as f:
    yaml.dump(data, f)
