# Imports
import snntorch as snn
from snntorch import surrogate, spikegen
from snntorch import functional as SF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

# Data Generator setup
datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train = datagen.flow_from_directory('balanced/train/', target_size=(224, 224), class_mode='binary', batch_size=64)
test = datagen.flow_from_directory('balanced/test/', target_size=(224, 224), class_mode='binary', batch_size=64)

# Convert TensorFlow ImageDataGenerator batches to PyTorch Tensors
def convert_to_tensorflow_tensor(generator):
    imgs, labels = next(generator)
    imgs_tensor = torch.tensor(imgs).permute(0, 3, 1, 2)  # Convert images to PyTorch tensor and permute to (N, C, H, W)
    labels_tensor = torch.tensor(labels).long()  # Convert labels to PyTorch tensor
    return imgs_tensor, labels_tensor

# Get tensors for train and test
train_imgs_tensor, train_labels_tensor = convert_to_tensorflow_tensor(train)
test_imgs_tensor, test_labels_tensor = convert_to_tensorflow_tensor(test)

# Create PyTorch TensorDataset
train_dataset = TensorDataset(train_imgs_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_imgs_tensor, test_labels_tensor)

# Define the percentage of data to load
load_percentage = 0.9

# Function to create a subset of the dataset based on the desired percentage
def get_subset(dataset, percentage):
    subset_size = int(len(dataset) * percentage)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return indices

# Create balanced sampler
def create_balanced_sampler(dataset, indices):
    labels = [dataset[idx][1].item() for idx in indices]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[dataset[idx][1].item()] for idx in indices]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Create subset indices
train_indices = get_subset(train_dataset, load_percentage)
test_indices = get_subset(test_dataset, load_percentage)

# Create samplers
train_sampler = create_balanced_sampler(train_dataset, train_indices)
test_sampler = create_balanced_sampler(test_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=train.batch_size, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=test.batch_size, sampler=test_sampler)

# Neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=15)
beta = 0.9
STEP_SIZE_TRAIN = 50

# Define SNN Model
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.drop1 = nn.Dropout(0.5)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        mem1, mem2, mem3, mem4 = [self.lif1.init_leaky() for _ in range(4)]

        for _ in range(STEP_SIZE_TRAIN):
            x_t = x.mean(dim=0)
            cur1 = F.max_pool2d(self.conv1(x_t), kernel_size=2)
            spk1, mem1 = self.lif1(self.bn1(cur1), mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), kernel_size=2)
            spk2, mem2 = self.lif2(self.bn2(cur2), mem2)

            cur3 = F.max_pool2d(self.conv3(spk2), kernel_size=2)
            spk3, mem3 = self.lif3(self.bn3(cur3), mem3)

            cur4 = self.drop1(self.fc1(spk3.view(spk3.size(0), -1)))
            spk4, mem4 = self.lif4(cur4, mem4)

        return self.fc2(spk4)

# Model, optimizer, and loss function setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SNNModel().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
loss_fn = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)


# Visualization Function
# Updated Plotting Function using plt.subplot
def visualize_predictions(loader, model):
    model.eval()
    data, labels = next(iter(loader))  # Get a batch of data and labels
    data = data.to(device)
    labels = labels.cpu().numpy()
    spk_data = spikegen.rate(data, num_steps=STEP_SIZE_TRAIN)

    with torch.no_grad():
        outputs = model(spk_data).sigmoid().cpu().numpy()
        preds = (outputs > 0.5).astype(int)

    num_images = len(preds)
    cols = 4  # Number of columns in the grid
    rows = (num_images + cols - 1) // cols  # Calculate number of rows
    plt.figure(figsize=(15, rows * 4))  # Adjust the figure size based on rows and columns

    for i in range(num_images):
        img = data[i].cpu().numpy().transpose(1, 2, 0)
        plt.subplot(rows, cols, i + 1)  # Add a subplot at the correct grid position
        plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1] == 1 else None)
        plt.title(f"Prediction: {preds[i][0]}, Label: {int(labels[i])}")
        plt.axis("off")

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


# Plotting Function
def plot_loss_accuracy(loss_hist, train_acc_hist, test_acc_hist):
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(test_acc_hist, label="Test Accuracy")
    plt.plot(train_acc_hist, label="Train Accuracy", linestyle='--')
    plt.legend()
    plt.title("Test & Train Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


# Initialize metrics and lists
num_epochs = 10
loss_hist, train_acc_hist, test_acc_hist = [], [], []
all_preds, all_labels = [], []

# Training loop
for epoch in range(num_epochs):
    net.train()
    correct_train, total_train, epoch_loss = 0, 0, 0

    for data, targets in train_loader:
        data = spikegen.rate(data.to(device), num_steps=STEP_SIZE_TRAIN)
        targets = targets.to(device).unsqueeze(1).float()

        outputs = net(data)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        preds = (outputs.sigmoid() > 0.5).float()
        correct_train += (preds == targets).sum().item()
        total_train += targets.size(0)

    train_acc = 100 * correct_train / total_train
    train_acc_hist.append(train_acc)
    loss_hist.append(epoch_loss / len(train_loader))

    # Validation
    net.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = spikegen.rate(data.to(device), num_steps=STEP_SIZE_TRAIN)
            targets = targets.to(device).unsqueeze(1).float()

            outputs = net(data)
            preds = (outputs.sigmoid() > 0.5).float()
            correct_test += (preds == targets).sum().item()
            total_test += targets.size(0)

            # Collect predictions for metrics
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(targets.cpu().numpy().flatten())

    test_acc = 100 * correct_test / total_test
    test_acc_hist.append(test_acc)
    scheduler.step(epoch_loss / len(train_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    # Visualization after training
    if epoch == num_epochs - 1:
        visualize_predictions(test_loader, net)
        
# Plotting loss and accuracy over epochs
plot_loss_accuracy(loss_hist, train_acc_hist, test_acc_hist)


    


# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
