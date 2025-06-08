import os

image_dir = "datasets/car-number-plate/images"
label_dir = "datasets/car-number-plate/labels"

image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Print paths for debugging
print("Image Files:", image_files)
print("Label Files:", label_files)

for image in image_files:
    # Get the name without extension and strip spaces
    image_name = os.path.splitext(image)[0].strip()

    # Construct the full paths and strip spaces from label filenames as well
    label_file = f"{image_name}.txt".strip()

    print(f"Checking {image} with label {label_file}")
    
    # Check if a corresponding label file exists
    if label_file not in label_files:
        print(f"Warning: Label file for {image_name} not found!")
    else:
        print(f"Label file for {image_name} exists.")
