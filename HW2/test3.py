#%%
import os
import medmnist
from medmnist import INFO
from torchvision.transforms import ToPILImage, Grayscale, ToTensor
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import deeptrack as dt
import torch
import deeplay as dl
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from matplotlib.patches import Rectangle


# Pick dataset
data_flag = 'pathmnist'  # change to other medmnist datasets if needed
download_dir = "MedMNIST_dataset"
os.makedirs(download_dir, exist_ok=True)
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Prepare transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    ToPILImage(),
    Grayscale(num_output_channels=1),
    ToTensor()
])

# Download datasets
train_dataset = DataClass(split='train', download=True, root=download_dir, transform=transform)
test_dataset = DataClass(split='test', download=True, root=download_dir, transform=transform)

# Save images in folders
def save_images(dataset, split):
    split_dir = os.path.join(download_dir, data_flag, split)
    os.makedirs(split_dir, exist_ok=True)

    if len(os.listdir(split_dir)) > 0:
        print(f"Skipping saving '{split}' images — already exists.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, (img, label) in enumerate(loader):
        save_path = os.path.join(split_dir, f"{idx:05d}_{label.item()}.png")
        save_image(img, save_path)

save_images(train_dataset, 'train')
save_images(test_dataset, 'test')
#%%
data_dir = os.path.join("MedMNIST_dataset", "pathmnist")
train_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "train"))
test_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "test"))
files = dt.sources.Join(train_files, test_files)

print(f"Number of train images: {len(train_files)}")
print(f"Number of test images: {len(test_files)}")

image_pip = (
    dt.LoadImage(files.path)
    >> (lambda x: x[..., 0:1]) 
    >> dt.NormalizeMinMax() 
    >> dt.MoveAxis(2, 0)  # HWC -> CHW
    >> dt.pytorch.ToTensor(dtype=torch.float)
)
vae = dl.VariationalAutoEncoder(
    input_size=[28,28],
    latent_dim=2, channels=[16,16],
    reconstruction_loss=torch.nn.BCELoss(reduction="sum"), beta=1,
).create()

print(vae)
train_dataset = dt.pytorch.Dataset(image_pip & image_pip, inputs=train_files)
train_loader = dl.DataLoader(train_dataset, batch_size=64, shuffle=True)
#%%
vae_trainer = dl.Trainer(max_epochs=10, accelerator="auto")
vae_trainer.fit(vae, train_loader)
#%%
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
#%%
label_pip = dt.Value(files.label_name[0]) >> int
test_dataset = dt.pytorch.Dataset(image_pip & label_pip, inputs=test_files)
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