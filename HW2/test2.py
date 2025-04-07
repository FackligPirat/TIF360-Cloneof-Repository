#%% Import and load/process data
from medmnist import PathMNIST
import os
import deeptrack as dt
import torch
import numpy as np
import deeplay as dl
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from PIL import Image
from deeptrack.sources import Source, Join
from matplotlib.patches import Rectangle
import glob
#%%
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

def rgb_to_grayscale(tensor):
    r, g, b = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

# Load datasets
PathMNIST_train_dataset = PathMNIST(split="train", download=True)
PathMNIST_test_dataset = PathMNIST(split="test", download=True)

# Convert to tensors and get labels
train_imgs = torch.tensor(PathMNIST_train_dataset.imgs).float()
test_imgs = torch.tensor(PathMNIST_test_dataset.imgs).float()
train_labels = torch.tensor(PathMNIST_train_dataset.labels).squeeze()
test_labels = torch.tensor(PathMNIST_test_dataset.labels).squeeze()

# Convert to grayscale
train_gray = torch.stack([rgb_to_grayscale(img) for img in train_imgs])
test_gray = torch.stack([rgb_to_grayscale(img) for img in test_imgs])

# Function to sort by label and save
def save_sorted_images(images, labels, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    
    # Sort images by label
    sorted_indices = torch.argsort(labels)
    sorted_images = images[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Save with new naming convention that reflects the sorting
    if not os.listdir(folder_path):
        for i, (img_tensor, label) in enumerate(zip(sorted_images, sorted_labels)):
            img = Image.fromarray(img_tensor.numpy().astype('uint8'))
            # Save with label prefix to maintain order
            img.save(os.path.join(folder_path, f"{label.item()}_{i}.png"))
    else:
        print(f"Images already exist in {folder_path}, skipping saving")

# Save sorted images
save_sorted_images(train_gray, train_labels, "pathmnist/train2")
save_sorted_images(test_gray, test_labels, "pathmnist/test2")

# Create DeepTrack sources
train_gray_source = dt.sources.ImageFolder(root="pathmnist/train2")
test_gray_source = dt.sources.ImageFolder(root="pathmnist/test2")
files = dt.sources.Join(train_gray_source, test_gray_source)

#%% Show the images

labels = PathMNIST_train_dataset.labels.squeeze()  # Ensure shape (N,)
unique_labels = np.unique(labels)
num_classes = len(unique_labels)

# Create figure: 2 examples × (RGB + Grayscale) per class
fig, axs = plt.subplots(
    nrows=num_classes,  # One row per class
    ncols=4,            # 4 columns: [RGB1, Gray1, RGB2, Gray2]
    figsize=(20, num_classes * 3)
)

# For each class, plot 2 examples (RGB + Grayscale pairs)
for row_idx, label in enumerate(unique_labels):
    # Get first two indices for this label
    indices = np.where(labels == label)[0][:2]  
    
    for col_offset, img_idx in enumerate(indices):
        # RGB (left)
        ax_rgb = axs[row_idx, 2 * col_offset]
        ax_rgb.imshow(train_imgs[img_idx].numpy().astype('uint8'))
        ax_rgb.set_title(f"Label {label}: RGB (Ex {col_offset+1})", fontsize=12)
        ax_rgb.axis('off')
        
        # Grayscale (right)
        ax_gray = axs[row_idx, 2 * col_offset + 1]
        ax_gray.imshow(train_gray[img_idx], cmap='Greys')
        ax_gray.set_title(f"Label {label}: Grayscale (Ex {col_offset+1})", fontsize=12)
        ax_gray.axis('off')

plt.suptitle("RGB vs. Grayscale Comparison (2 Examples per Class)", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
#%% Show the images 2
original_rgb = train_imgs[:10]  # Shape: [10, 28, 28, 3]

# Load FIRST 10 generated grayscale images from disk (enforce order)
gray_paths = [f"pathmnist/train2/0_{i}.png" for i in range(10)]  # Assumes filenames are image_0.png, image_1.png, etc.
generated_gray = [np.array(Image.open(path)) for path in gray_paths]

# Plot side-by-side
fig, axs = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    # Original RGB (top row)
    axs[0, i].imshow(original_rgb[i].numpy().astype('uint8'))
    axs[0, i].set_title(f"Original {i}")
    axs[0, i].axis('off')
    
    # Generated grayscale (bottom row)
    axs[1, i].imshow(generated_gray[i], cmap='gray')
    axs[1, i].set_title(f"Generated {i}")
    axs[1, i].axis('off')

plt.suptitle("Original RGB vs. Generated Grayscale (First 10 Ordered Pairs)", y=1.05)
plt.tight_layout()
plt.show()
# %% Create VAE and pipeline
image_pip = (dt.LoadImage(files.path) >> dt.NormalizeMinMax()
             >> dt.MoveAxis(2, 0) >> dt.pytorch.ToTensor(dtype=torch.float))

vae = dl.VariationalAutoEncoder(
    latent_dim=2, channels=[32,32],
    reconstruction_loss=torch.nn.BCELoss(reduction="sum"), beta=1,
).create()

train_dataset = dt.pytorch.Dataset(image_pip & image_pip, inputs=train_gray_source)
train_loader = dl.DataLoader(train_dataset, batch_size=64, shuffle=True)
#%% Training
#vae = vae.to(device)
vae_trainer = dl.Trainer(max_epochs=10, accelerator="auto")
vae_trainer.fit(vae, train_loader)

# %% Generate images
img_num, img_size = 21, 28
z0_grid = z1_grid = Normal(0, 1).icdf(torch.linspace(0.001, 0.999, img_num))

image = np.zeros((img_num * img_size, img_num * img_size))
for i0, z0 in enumerate(z0_grid):
    for i1, z1 in enumerate(z1_grid):
      z = torch.stack((z0, z1)).unsqueeze(0)
      generated_image = vae.decode(z).clone().detach()
      image[i1 * img_size : (i1 + 1) * img_size,
            i0 * img_size : (i0 + 1) * img_size] = \
          generated_image.numpy().squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.xlabel("z0", fontsize=24)
plt.xticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size),
           np.round(z0_grid.numpy(), 1))
plt.ylabel("z1", fontsize=24)
plt.yticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size),
           np.round(z1_grid.numpy(), 1))
plt.show()
#%% Latent space
label_pip = dt.Value(files.label_name[0]) >> int
test_dataset = dt.pytorch.Dataset(image_pip & label_pip, inputs=test_gray_source)
test_loader = dl.DataLoader(test_dataset, batch_size=64, shuffle=False)
mu_list, test_labels = [], []
for image, label in test_loader:
    mu, _ = vae.encode(image)
    mu_list.append(mu)
    test_labels.append(label)
mu_array = torch.cat(mu_list, dim=0).detach().numpy()
test_labels = torch.cat(test_labels, dim=0).numpy()
plt.figure(figsize=(12, 10))
plt.scatter(mu_array[:, 0], mu_array[:, 1], s=3, c=test_labels, cmap="tab10")
plt.gca().add_patch(Rectangle((-3.1, -3.1), 6.2, 6.2, fc="none", ec="k", lw=1))
plt.xlabel("mu_array[:, 0]", fontsize=24)
plt.ylabel("mu_array[:, 1]", fontsize=24)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.colorbar()
plt.show()
# %%
