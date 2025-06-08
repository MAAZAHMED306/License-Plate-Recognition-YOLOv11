import os
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

# Paths for the dataset and pre-trained model weights
root_dir = "C:/Users/MAAZ/Source-Code/datasets"
train_images = f"{root_dir}/images/train"
valid_images = f"{root_dir}/images/valid"
test_images = f"{root_dir}/images/test"

train_labels = f"{root_dir}/labels/train"
valid_labels = f"{root_dir}/labels/valid"
test_labels = f"{root_dir}/labels/test"

yaml_file = "C:/Users/MAAZ/Source-Code/number-plate.yaml"  # Path to the .yaml file

# Check if the dataset YAML exists
if not Path(yaml_file).exists():
    print(f"YAML file '{yaml_file}' does not exist. Please check the path.")
    exit(1)

# Load the dataset YAML configuration
with open(yaml_file, 'r') as file:
    dataset_config = yaml.safe_load(file)

# Ensure that 'train', 'val', and 'test' directories are correctly set in YAML
dataset_config["train"] = train_images
dataset_config["val"] = valid_images
dataset_config["test"] = test_images

# Specify the pre-trained model to use (using YOLOv11 weights)
pretrained_model = "C:/Users/MAAZ/Source-Code/yolo11s.pt"  # Path to the pre-trained YOLOv11 weights

if not Path(pretrained_model).exists():
    print(f"Pre-trained model weight file '{pretrained_model}' does not exist. Please check the path.")
    exit(1)

# Check for GPU availability and print whether it's being used
def check_gpu():
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU is available. Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU is not available. Using CPU instead.")
    return gpu_available

# Start the YOLOv11 training
def train_yolov11():
    # Check if GPU is available
    check_gpu()

    # Load the pre-trained YOLOv11 model (ensure it's using the correct model path)
    model = YOLO(pretrained_model)  # Load pre-trained YOLOv11 model

    # Train the model
    model.train(
        data=yaml_file,  # Use the path to the dataset yaml file, not the dictionary
        epochs=50,       # Adjust the number of epochs
        batch=16,        # Adjust the batch size
        imgsz=640,       # Image size for training (adjust as needed)
        device="0" if torch.cuda.is_available() else "cpu",  # Use GPU if available, otherwise use CPU
        project="runs/train", # Directory where training results will be saved
        name="number-plate",  # Name of the experiment
        exist_ok=True         # Allow overwriting the existing folder with new results
    )

    print("Training completed successfully!")
    return model

# Wrap the training call in the __main__ block to avoid multiprocessing issues in Windows
if __name__ == '__main__':
    # Call the train function
    trained_model = train_yolov11()

    # Save the final model weights after training
    trained_model.save("number_plate_trained_weights.pt")
    print("Model weights saved as 'number_plate_trained_weights.pt'.")
