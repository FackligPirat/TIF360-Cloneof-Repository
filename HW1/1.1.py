#%%
import requests
import os
from io import BytesIO
from zipfile import ZipFile
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import torchmetrics as tm

#%% Load MNIST data
if not os.path.exists("MNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/MNIST_dataset")

train_path = os.path.join("MNIST_dataset", "mnist", "train")
train_image_files = sorted(os.listdir(train_path))
test_path = os.path.join("MNIST_dataset", "mnist", "test")
test_image_files = sorted(os.listdir(test_path))

#%% Dataset and transforms
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

transform = Compose([ToTensor()])

# Prepare training data
train_images = [plt.imread(os.path.join(train_path, file)) for file in train_image_files]
train_digits = [int(os.path.basename(file)[0]) for file in train_image_files]
train_dataset = MNISTDataset(train_images, train_digits, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Prepare test data
test_images = [plt.imread(os.path.join(test_path, file)) for file in test_image_files]
test_digits = [int(os.path.basename(file)[0]) for file in test_image_files]
test_dataset = MNISTDataset(test_images, test_digits, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#%% Model definition
conv_base = dl.ConvolutionalNeuralNetwork(
    in_channels=1, #Grey scale
    hidden_channels=[16, 16, 32],
    out_channels=32,
)
conv_base.blocks[2].pool.configure(torch.nn.MaxPool2d, kernel_size=2)

connector = dl.Layer(torch.nn.AdaptiveAvgPool2d, output_size=1)

dense_top = dl.MultiLayerPerceptron(
    in_features=32,
    hidden_features=[],
    out_features=10, # Since 10 digits
    out_activation=torch.nn.Identity,
)

cnn = dl.Sequential(conv_base, connector, dense_top)

cnn_classifier = dl.Classifier(
    model=cnn,
    optimizer=dl.RMSprop(lr=0.001),
    loss=torch.nn.CrossEntropyLoss(),
    metrics=[tm.Accuracy(task="multiclass", num_classes=10)],
).create()

#%% Training
cnn_trainer = dl.Trainer(max_epochs=5, accelerator="auto")
cnn_trainer.fit(cnn_classifier, train_loader)

#%% Test
test_results = cnn_trainer.test(cnn_classifier, test_loader)

#%% Plotting ROC
def plot_multiclass_roc(classifier, loader, num_classes=10):
    roc = tm.ROC(task="multiclass", num_classes=num_classes)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            preds = classifier(images)
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    roc.update(all_preds, all_labels)
    
    fig, ax = roc.plot()
    ax.set_title('Multi-class ROC Curves')
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.show()

plot_multiclass_roc(cnn_classifier, test_loader)

# %%
