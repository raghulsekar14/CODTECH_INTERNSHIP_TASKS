import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data augmentation and normalization for training data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),          
    transforms.RandomCrop(32, padding=4),        
    transforms.ToTensor(),                       
    transforms.Normalize((0.5, 0.5, 0.5),        
                         (0.5, 0.5, 0.5))
])

# Normalization for test data (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),                       
    transforms.Normalize((0.5, 0.5, 0.5),      
                         (0.5, 0.5, 0.5))
])

# Paths for training and testing data
train_path = os.path.join(BASE_DIR, "Task_3", "train")
test_path = os.path.join(BASE_DIR, "Task_3", "test")

# Load CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=train_path,
    train=True,
    download=True,
    transform=transform_train
)

# Load CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root=test_path,
    train=False,
    download=True,
    transform=transform_test
)

# DataLoader for batching and shuffling training data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# DataLoader for batching test data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class names in CIFAR-10 dataset
class_names = train_dataset.classes

# Define the Convolutional Neural Network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Feature extraction layers (convolution + pooling)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # First convolution layer
            nn.ReLU(),                        # Activation function
            nn.MaxPool2d(2),                  # Downsampling
            nn.Conv2d(32, 64, 3, padding=1),  # Second convolution layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), # Third convolution layer
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),      # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout for regularization
            nn.Linear(256, 10)                # Output layer for 10 classes
        )

    # Forward pass of the model
    def forward(self, x):
        x = self.features(x)                  # Extract features
        x = x.view(x.size(0), -1)             # Flatten feature maps
        return self.classifier(x)             # Classify features

# Initialize the CNN model and move it to selected device
model = CNN().to(device)

# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer for updating model weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()                             # Set model to training mode
    correct = 0
    total = 0

    # Iterate over training batches
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                 # Clear previous gradients
        outputs = model(images)               # Forward pass
        loss = criterion(outputs, labels)     # Compute loss
        loss.backward()                       # Backpropagation
        optimizer.step()                      # Update weights

        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)               # Total samples
        correct += (predicted == labels).sum().item()  # Correct predictions

    # Print training accuracy for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] Accuracy: {100 * correct / total:.2f}%")

# Evaluation phase
model.eval()                                  # Set model to evaluation mode
correct = 0
total = 0
all_preds = []
all_labels = []

# Disable gradient computation for testing
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)               # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predicted class

        total += labels.size(0)               # Total samples
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())  # Store predictions
        all_labels.extend(labels.cpu().numpy())    # Store true labels

# Print test accuracy
print("\nTest Accuracy:", 100 * correct / total)

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save trained model weights
torch.save(model.state_dict(), "cnn_model.pth")
print("\nModel saved as cnn_model.pth")
