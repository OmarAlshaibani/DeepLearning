import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import shap

# Initialize lists for images and labels
images = []
labels = []  # Classification labels (1: Drone)
annotations = []  # Bounding box annotations for drones

# Path to drone dataset
drone_dataset_path = DL\Eye_Project\Drone\dataset_txt"

# Load Drone dataset
for image_file in glob.glob(os.path.join(drone_dataset_path, "*.jpg")):
    try:
        # Load and preprocess image
        img = Image.open(image_file).convert('RGB').resize((256, 256))
        images.append(np.asarray(img))
        labels.append(1)  # Label: 1 for Drone

        # Load bounding box annotation
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        with open(annotation_file, 'r') as f:
            bbox = list(map(float, f.readline().split()[1:]))
            annotations.append(bbox)
    except Exception as e:
        print(f"Error loading drone data: {e}")

# Convert data to numpy arrays
images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels, dtype='float32')
annotations = np.array(annotations, dtype='float32')

# Print shapes to verify
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Annotations shape:", annotations.shape)

# Split data into training and testing sets
(train_images, test_images, train_labels, test_labels,
 train_bboxes, test_bboxes) = train_test_split(images, labels, annotations, test_size=0.1, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)


# Model architecture
def build_model():
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                                             input_tensor=tf.keras.Input(shape=(256, 256, 3)))
    base_model.trainable = True

    # Shared feature extraction
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout for regularization

    # Classification head
    class_head = tf.keras.layers.Dense(16, activation="relu")(x)
    class_head = tf.keras.layers.Dropout(0.3)(class_head)
    class_head = tf.keras.layers.Dense(1, activation="sigmoid", name="classification")(class_head)

    # Bounding box regression head
    bbox_head = tf.keras.layers.Dense(32, activation="relu")(x)
    bbox_head = tf.keras.layers.Dense(16, activation="relu")(bbox_head)
    bbox_head = tf.keras.layers.Dense(4, activation="linear", name="bounding_box")(bbox_head)

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_head, bbox_head])
    return model


model = build_model()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={"classification": "binary_crossentropy", "bounding_box": "mse"},
              metrics={"classification": "accuracy", "bounding_box": "mse"})

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    datagen.flow(train_images, {"classification": train_labels, "bounding_box": train_bboxes}, batch_size=32),
    validation_data=(test_images, {"classification": test_labels, "bounding_box": test_bboxes}),
    epochs=100,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Evaluate the model
loss, class_loss, bbox_loss, class_acc, bbox_mse = model.evaluate(
    test_images, {"classification": test_labels, "bounding_box": test_bboxes}, verbose=1
)
print(f"Test Classification Accuracy: {class_acc:.4f}")
print(f"Test Bounding Box MSE: {bbox_mse:.4f}")

# Classification Report
pred_classes, pred_bboxes = model.predict(test_images)
pred_labels = (pred_classes > 0.5).astype("int32").flatten()
print("\nClassification Report:")
print(classification_report(test_labels, pred_labels, target_names=["Not Drone", "Drone"]))

# Confusion Matrix
cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Drone", "Drone"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.close()

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['classification_accuracy'], label="Train Accuracy")
plt.plot(history.history['val_classification_accuracy'], label="Validation Accuracy")
plt.title("Classification Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['classification_loss'], label="Train Loss")
plt.plot(history.history['val_classification_loss'], label="Validation Loss")
plt.title("Classification Loss")
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_loss_curves.png')
plt.close()


# Function to visualize model predictions
def visualize_prediction(image, true_label, pred_label, true_bbox, pred_bbox):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot true bounding box
    rect = plt.Rectangle((true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
                         fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect)

    # Plot predicted bounding box
    rect = plt.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3],
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    plt.title(
        f"True: {'Drone' if true_label == 1 else 'Not Drone'}, Predicted: {'Drone' if pred_label == 1 else 'Not Drone'}")
    plt.axis('off')
    return fig


# Visualize some predictions
num_visualizations = 5
for i in range(num_visualizations):
    fig = visualize_prediction(test_images[i],
                               test_labels[i],
                               pred_labels[i],
                               test_bboxes[i],
                               pred_bboxes[i])
    fig.savefig(f'prediction_visualization_{i + 1}.png')
    plt.close(fig)

print("All visualizations and results have been saved.")

# Model interpretation using SHAP
explainer = shap.DeepExplainer(model, train_images[:100])
shap_values = explainer.shap_values(test_images[:10])

# Plot SHAP values for classification
shap.image_plot(shap_values[0], -test_images[:10])
plt.savefig('shap_classification.png')
plt.close()

print("SHAP visualization saved.")

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

print("Model architecture saved.")


# Hyperparameter tuning using random search
def create_model(learning_rate=0.001):
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss={"classification": "binary_crossentropy", "bounding_box": "mse"},
                  metrics={"classification": "accuracy", "bounding_box": "mse"})
    return model


param_dist = {
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64, 128],
    'epochs': [50, 100, 150]
}

random_search = RandomizedSearchCV(
    estimator=KerasClassifier(build_fn=create_model),
    param_distributions=param_dist,
    n_iter=10,
    cv=3
)

random_search_result = random_search.fit(train_images, train_labels)
print("Best parameters:", random_search_result.best_params_)

print("Project completed successfully.")
