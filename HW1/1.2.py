#%% 1.1
import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette, heatmap
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
import deeplay as dl
import torchmetrics as tm
from skimage.transform import resize
from skimage.exposure import rescale_intensity

# Load MNIST data
if not os.path.exists("MNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/MNIST_dataset")

train_path = os.path.join("MNIST_dataset", "mnist", "train")
train_image_files = sorted(os.listdir(train_path))
test_path = os.path.join("MNIST_dataset", "mnist", "test")
test_image_files = sorted(os.listdir(test_path))

# Dataset with one-hot encoding
class MNISTOneHotDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_classes = 10  # MNIST has 10 classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to one-hot encoding
        one_hot_label = torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).float()
        
        return image, one_hot_label

transform = Compose([ToTensor()])

# Prepare training data
train_images = [plt.imread(os.path.join(train_path, file)) for file in train_image_files] #List of numpy array
train_digits = [int(os.path.basename(file)[0]) for file in train_image_files]
train_dataset = MNISTOneHotDataset(train_images, train_digits, transform=transform) #Makes into dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #Wraps into loader and shuffles data

# Prepare test data
test_images = [plt.imread(os.path.join(test_path, file)) for file in test_image_files]
test_digits = [int(os.path.basename(file)[0]) for file in test_image_files]
test_dataset = MNISTOneHotDataset(test_images, test_digits, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


conv_base = dl.ConvolutionalNeuralNetwork(
    in_channels=1,
    hidden_channels=[16, 16, 32],
    out_channels=32,
)
conv_base.blocks[2].pool.configure(torch.nn.MaxPool2d, kernel_size=2)

connector = dl.Layer(torch.nn.AdaptiveAvgPool2d, output_size=1)

dense_top = dl.MultiLayerPerceptron(
    in_features=32,
    hidden_features=[],
    out_features=10,
    out_activation=torch.nn.Softmax(dim=1), 
)

cnn = dl.Sequential(conv_base, connector, dense_top)

# Using categorical crossentropy loss since Softmax activation
cnn_classifier = dl.Classifier(
    model=cnn,
    optimizer=dl.RMSprop(lr=0.001),
    metrics=[tm.Accuracy(task="multilabel", num_labels=10)],
).create()

#%% Training and testing 1.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # If cuda available
print(f"Using device: {device}")

# Move your model to the GPU
cnn_classifier = cnn_classifier.to(device)

cnn_trainer = dl.Trainer(max_epochs=5, accelerator="auto", devices = "auto")
cnn_trainer.fit(cnn_classifier, train_loader)

# Test
test_results = cnn_trainer.test(cnn_classifier, test_loader)

#%% 1.2

# --- Grad-CAM Function for Multi-Class ---
def grad_cam_multiclass(model, image, target_class=None, layer=None):
    """
    Compute Grad-CAM heatmap for a multi-class model.
    
    Args:
        model: Trained PyTorch model.
        image: Input image (C, H, W).
        target_class: Class index to explain (if None, uses predicted class).
        layer: Target convolutional layer (default: last conv layer).
    """
    # Register hooks
    hookdata = {}
    
    def fwd_hook(layer, input, output):
        hookdata["activations"] = output.detach()
    
    def bwd_hook(layer, grad_input, grad_output):
        hookdata["gradients"] = grad_output[0].detach()
    
    # Default to last convolutional layer if not specified
    if layer is None:
        layer = model[0].blocks[-1].layer  # Assuming conv_base is model[0]
    
    # Register hooks
    handle_fwd = layer.register_forward_hook(fwd_hook)
    handle_bwd = layer.register_full_backward_hook(bwd_hook)
    
    # Forward pass
    image_tensor = image.unsqueeze(0)  # Add batch dim
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    
    # Use predicted class if target_class is None
    if target_class is None:
        target_class = torch.argmax(probs).item()
    
    # Zero gradients, then backward pass for target class
    model.zero_grad()
    logits[0, target_class].backward(retain_graph=True)
    
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Get activations and gradients
    activations = hookdata["activations"][0]  # Shape: (C, H, W)
    gradients = hookdata["gradients"][0]      # Shape: (C, H, W)
    
    # Compute Grad-CAM
    pooled_gradients = gradients.mean(dim=[1, 2], keepdim=True)
    heatmap = (pooled_gradients * activations).sum(dim=0)
    heatmap = torch.relu(heatmap).cpu().numpy()
    
    # Resize heatmap to match input image
    heatmap_resized = resize(heatmap, image.shape[1:], order=3)  # bicubic interpolation
    heatmap_rescaled = rescale_intensity(heatmap_resized, out_range=(0, 1))
    
    return heatmap_rescaled, target_class

# --- Visualization Function ---
def plot_gradcam(image, heatmap, target_class):
    """Plot original image, heatmap, and overlay."""
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.title(f"Input (True: {true_class})", fontsize=12)
    plt.axis("off")
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="viridis")
    plt.title(f"Grad-CAM (Class: {target_class})", fontsize=12)
    plt.axis("off")
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.imshow(heatmap, cmap="viridis", alpha=0.5)
    plt.title("Overlay", fontsize=12)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# 1. Pick a test image
test_image, true_label_one_hot = next(iter(test_loader))
true_class = torch.argmax(true_label_one_hot[0]).item()
image = test_image[0]  # Shape: (1, 28, 28)

# 2. Compute Grad-CAM for the predicted class
heatmap, pred_class = grad_cam_multiclass(cnn_classifier.model, image)

# 3. Visualize
plot_gradcam(image, heatmap, pred_class)
# %%
