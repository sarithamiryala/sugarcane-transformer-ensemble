# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm

# !pip install split-folders

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import splitfolders

input_folder = "/kaggle/input/sugarcane-plant-diseases-dataset/Sugarcane_leafs"
splitfolders.ratio(input_folder, output="/kaggle/working/sugarcanedisease", seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (256, 256)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   zoom_range=0.2)

val_test_datagen = ImageDataGenerator(rescale=1./255)


#----------------------------------------

train_generator = train_datagen.flow_from_directory('/kaggle/working/sugarcanedisease/train',
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

val_generator = val_test_datagen.flow_from_directory('/kaggle/working/sugarcanedisease/val',
                                                target_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory('/kaggle/working/sugarcanedisease/test',
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=False)

import os
import torch
from torchvision import datasets, transforms

# Define data directories
data_dir = '/kaggle/working/sugarcanedisease'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define data transformations
# Corresponding to Keras ImageDataGenerator parameters
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        # No need for rescale=1./255 because ToTensor() does that
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # No need for rescale=1./255 because ToTensor() does that
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # No need for rescale=1./255 because ToTensor() does that
    ]),
}

# Create datasets using ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Print dataset sizes
print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

train_dataset.classes


# Load the feature extractor and the model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(train_dataset.classes))

# Clear GPU cache to free up memory
torch.cuda.empty_cache()

# Reduce batch size in your dataloaders (e.g., to 16 or 8)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Move the model to the GPU if available
model.to(device)

# Print the model architecture
print(model)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
num_epochs = 10

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # If your model returns logits directly, you can use outputs = model(inputs)
        # If your model returns a dictionary, adjust accordingly
        outputs = model(inputs).logits if hasattr(model(inputs), 'logits') else model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # **Fixed print statement with matching quotation marks**
    print(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).logits if hasattr(model(inputs), 'logits') else model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f'Validation - Epoch {epoch+1}/{num_epochs}, Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}')

    # Step the scheduler
    scheduler.step()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Collect all the true labels and predicted labels for the validation set
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).logits
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Function to show an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of validation data
inputs, classes = next(iter(val_loader))

# Make predictions
outputs = model(inputs.to(device)).logits
_, preds = torch.max(outputs, 1)

# Plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(4):
    ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
    imshow(inputs.cpu().data[idx])
    ax.set_title(f"Predicted: {train_dataset.classes[preds[idx]]}\nTrue: {train_dataset.classes[classes[idx]]}")

# Save the model
torch.save(model.state_dict(), '/kaggle/working//vit_sugarcane_disease_detection.pth')

import matplotlib.pyplot as plt
import numpy as np
import torch

# Function to show an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    # Unnormalize if necessary
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide axes for clarity

# Get a batch of validation data
inputs, classes = next(iter(val_loader))

# Move inputs and labels to the device
inputs = inputs.to(device)
classes = classes.to(device)

# Set model to evaluation mode
model.eval()

# Make predictions without tracking gradients
with torch.no_grad():
    # Obtain the outputs from the model
    outputs = model(inputs)

    # **Extract the logits Tensor from the outputs**
    outputs = outputs.logits  # Access the 'logits' attribute

    # Get the predicted classes
    _, preds = torch.max(outputs, 1)

# Move data to CPU for visualization
inputs = inputs.cpu()
classes = classes.cpu()
preds = preds.cpu()

# Get class names
class_names = val_dataset.classes  # Use validation dataset's classes

# Number of images to display
num_images = 4

# Plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(15, 6))
for idx in range(num_images):
    ax = fig.add_subplot(1, num_images, idx+1)
    imshow(inputs[idx])

    # Get class names using class indices
    pred_class = class_names[preds[idx]]
    true_class = class_names[classes[idx]]

    # Set the title with predicted and true labels
    ax.set_title(f"Predicted: {pred_class}\nTrue: {true_class}")
    ax.axis('off')  # Hide axes ticks

plt.tight_layout()
plt.show()

