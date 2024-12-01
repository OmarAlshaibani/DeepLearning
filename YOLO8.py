import glob
import os
import shutil

from matplotlib import pyplot as plt
from tqdm import tqdm

DATA_DIR = r"C:\Users\pythonProject\DL\Eye_Project\Drone\dataset_txt"

images = []
labels = []
# Get a list of all image files in the directory
image_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))  # Adjust the extension if necessary


for image_path in tqdm(image_files):     #slicing for insufficient memory
    images.append(image_path)
    label_path = image_path.split('.')[0] + '.txt'
    labels.append(label_path)


from sklearn.model_selection import train_test_split
split = train_test_split(images, labels, test_size=0.10, random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]

TRAIN_IMAGE_DIR = 'train/images'
TRAIN_LABEL_DIR = 'train/labels'
VAL_IMAGE_DIR = 'valid/images'
VAL_LABEL_DIR = 'valid/labels'

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

for path in tqdm(trainImages):
    shutil.copyfile(path, os.path.join(TRAIN_IMAGE_DIR, os.path.basename(path)))

for path in tqdm(testImages):
    shutil.copyfile(path, os.path.join(VAL_IMAGE_DIR, os.path.basename(path)))

for path in tqdm(trainTargets):
    shutil.copyfile(path, os.path.join(TRAIN_LABEL_DIR, os.path.basename(path)))

for path in tqdm(testTargets):
    shutil.copyfile(path, os.path.join(VAL_LABEL_DIR, os.path.basename(path)))


# Disable wandb
import os
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO
# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model using your custom dataset
results = model.train(
    data='C:/Users/omara/OneDrive/Desktop/pythonProject/DL/Eye_Project/Drone/drone.yaml',
    imgsz=256,
    epochs=45,
    batch=8,
    name='yolov8s_v8_501e'
)

# Export the model
success = model.export(format="onnx")
print("Model exported:", success)

import math
import random


# Plot and visualize images in a 2x2 grid.
def visualize(result_dir, num_samples=None, num_cols=1):
    """
    Function accepts a list of images and plots
    """
    image_names = sorted(glob.glob(os.path.join(result_dir, '*.jpg')))

    if num_samples is not None:
        image_names = random.sample(image_names, num_samples)

    num_images = len(image_names)
    num_rows = int(math.ceil(num_images / num_cols))
    plt.figure(figsize=(12 * num_cols, 6 * num_rows))

    for i, image_name in enumerate(image_names):
        image = plt.imread(image_name)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# results = model("/kaggle/input/drone-dataset-uav/drone_dataset_yolo/dataset_txt/0014.jpg", conf=0.5, agnostic_nms=True, iou=0.5)  # predict on an image
results = model("valid/images", conf=0.5, agnostic_nms=True, iou=0.5, save=True)
res_plotted = results[0].plot()
plt.imshow(res_plotted)
plt.show()

indices = list(range(len(results)))
random_indices = random.sample(indices, 10)
num_cols = 2
num_rows = 5

plt.figure(figsize=(12 * num_cols, 6 * num_rows))

for i, idx in enumerate(random_indices):
    image = results[i].plot()
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()
plt.show()

success = model.export(format="onnx")  # export the model to ONNX format
success


# Plot results (training/validation accuracy and loss curves)
def plot_training_results(results_dir):
    results_file = os.path.join(results_dir, "results.png")
    if not os.path.exists(results_file):
        print(f"No results found in {results_dir}")
        return
    metrics = plt.imread(results_file)
    plt.figure(figsize=(10, 6))
    plt.imshow(metrics)
    plt.axis('off')
    plt.title(f"Training Results: {results_dir}")
    plt.show()

# Visualize the training metrics
plot_training_results("runs/train/yolov8s_v8_501e")

# Visualize predictions on validation dataset
def visualize_predictions(model, data_dir, num_samples=10):
    results = model(data_dir, conf=0.5, agnostic_nms=True, iou=0.5, save=True)
    indices = random.sample(range(len(results)), min(num_samples, len(results)))
    num_cols = 2
    num_rows = math.ceil(len(indices) / num_cols)
    plt.figure(figsize=(12 * num_cols, 6 * num_rows))
    for i, idx in enumerate(indices):
        image = results[idx].plot()
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(model, VAL_IMAGE_DIR)

# Summary
print("Training Summary:")
print(f"  - Validation mAP: {results.metrics['box.map']}")
print(f"  - Training loss: {results.metrics['box.loss']}")

