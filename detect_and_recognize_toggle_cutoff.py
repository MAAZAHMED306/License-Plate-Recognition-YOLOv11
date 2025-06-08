# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import csv
import cv2
import torch
import difflib
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from easyocr import Reader

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"Using GPU: {gpu_name}")
    print(f"CUDA Version: {cuda_version}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Constants
CONFIDENCE_THRESHOLD = 0.6
COLOR = (0, 255, 0)
USE_CUTOFF_LINE = False  # Set to True to only detect plates near a horizontal line

# Input video/image path
VIDEO_PATH = "datasets/car-number-plate/videos/traffic.mp4"
IMAGE_PATH = "datasets/images/test/frame125.jpeg"

# Get timestamp for filename uniqueness
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Extract base filename of video (e.g., "traffic")
video_filename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
logic_suffix = "_with_cutoff" if USE_CUTOFF_LINE else "_no_cutoff"

# Generate output filenames based on logic and timestamp
OUTPUT_CSV = f"{video_filename}{logic_suffix}_{timestamp}.csv"
OUTPUT_VIDEO = f"{video_filename}{logic_suffix}_{timestamp}.mp4"

# Initialize YOLO model and EasyOCR
model = YOLO(r'C:\Users\MAAZ\Source-Code\runs\train\number-plate\weights\best.pt')
reader = Reader(['en'], gpu=True)

# Deduplication history
recent_detections = deque(maxlen=20)

# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------

def compute_iou(box1, box2):
    """Calculate Intersection-over-Union (IoU) for two bounding boxes"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1 = max(x1, a1)
    yi1 = max(y1, b1)
    xi2 = min(x2, a2)
    yi2 = min(y2, b2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def is_duplicate(new_box, new_text):
    """Check if a detected plate is a duplicate"""
    for prev_box, prev_text in recent_detections:
        iou = compute_iou(new_box, prev_box)
        similarity = difflib.SequenceMatcher(None, new_text, prev_text).ratio()
        if iou > 0.5 and similarity > 0.7:
            return True
    return False

def detect_number_plates(image, model):
    """Use YOLO to detect number plate bounding boxes"""
    start = time.time()
    detections = model.predict(image)[0].boxes.data
    number_plate_list = []

    if detections.shape != torch.Size([0, 6]):
        for det in detections:
            if float(det[4]) >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = map(int, det[:4])
                number_plate_list.append([[xmin, ymin, xmax, ymax]])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
        print(f"Detected {len(number_plate_list)} plate(s) in {(time.time() - start) * 1000:.0f} ms")
    else:
        print("No plates detected.")
    return number_plate_list

def recognize_number_plates(image, reader, number_plate_list):
    """Use OCR to read text from each detected number plate"""
    start = time.time()
    for i, box in enumerate(number_plate_list):
        xmin, ymin, xmax, ymax = box[0]
        cropped = image[ymin:ymax, xmin:xmax]
        results = reader.readtext(cropped, paragraph=True)
        text = results[0][1] if results else ""
        number_plate_list[i].append(text)
    print(f"OCR completed in {(time.time() - start) * 1000:.0f} ms")
    return number_plate_list

def save_to_csv(data, mode='w'):
    """Save detected plate data to CSV file"""
    with open(OUTPUT_CSV, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(["source", "box", "text"])
        for entry in data:
            box, text, source = entry
            writer.writerow([source, box, text])

# ----------------------------------------
# IMAGE PROCESSING
# ----------------------------------------

def process_image(image_path):
    """Run detection + OCR on a single image"""
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    plate_data = detect_number_plates(image, model)
    if plate_data:
        plate_data = recognize_number_plates(image, reader, plate_data)
        for item in plate_data:
            item.append(image_path)  # Add source path for CSV
        save_to_csv(plate_data)

        # Annotate and save output image
        for box, text, _ in plate_data:
            cv2.putText(image, text, (box[0], box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        output_image_path = image_path.replace(".jpeg", "_detected.jpeg")
        cv2.imwrite(output_image_path, image)
        cv2.imshow("Image Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Processed image saved to: {output_image_path}")

# ----------------------------------------
# VIDEO PROCESSING
# ----------------------------------------

def process_video(video_path):
    """Run detection + OCR on every frame of a video"""
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file not found or couldn't open.")
        return

    writer = None
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cutoff_line = int(frame_height * 0.7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        plate_data = detect_number_plates(frame, model)
        valid_plate_data = []

        for box in plate_data:
            xmin, ymin, xmax, ymax = box[0]
            box_center_y = (ymin + ymax) // 2

            # Apply or skip cutoff logic based on config
            if not USE_CUTOFF_LINE or abs(box_center_y - cutoff_line) < 20:
                cropped = frame[ymin:ymax, xmin:xmax]
                results = reader.readtext(cropped, paragraph=True)
                text = results[0][1] if results else ""

                if not is_duplicate(box[0], text):
                    recent_detections.append((box[0], text))
                    valid_plate_data.append([box[0], text, video_path])
                    cv2.putText(frame, text, (xmin, ymax + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR, 2)

        if valid_plate_data:
            save_to_csv(valid_plate_data, mode='a')

        # Draw cutoff line if enabled
        if USE_CUTOFF_LINE:
            cv2.line(frame, (0, cutoff_line), (frame_width, cutoff_line), (0, 0, 255), 2)

        # Display result and write to video
        cv2.imshow("Video Output", frame)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 25, (frame.shape[1], frame.shape[0]))
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to: {OUTPUT_VIDEO}")
    print(f"Results saved to CSV: {OUTPUT_CSV}")

# ----------------------------------------
# RUN
# ----------------------------------------

process_video(VIDEO_PATH)
# To test an image instead, use: 
# process_image(IMAGE_PATH)
