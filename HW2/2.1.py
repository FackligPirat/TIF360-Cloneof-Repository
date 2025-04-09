#%% Import libraries
import os
import deeptrack as dt
import torch
import torch.nn as nn
import deeplay as dl
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%load and create data
def plot_image(title, image):
    """Plot a grayscale image with a title."""
    plt.imshow(image, cmap="gray")
    plt.title(title, fontsize=30)
    plt.axis("off")
    plt.show()
    
particle = dt.Sphere(
    position=np.array([0.5, 0.5]) * 64, position_unit="pixel",
    radius=500 * dt.units.nm, refractive_index=1.45 + 0.02j,
)
brightfield_microscope = dt.Brightfield(
    wavelength=500 * dt.units.nm, NA=1.0, resolution=1 * dt.units.um,
    magnification=10, refractive_index_medium=1.33, upsample=2,
    output_region=(0, 0, 64, 64),
)
noise = dt.Poisson(snr=lambda: 5.0 + np.random.rand())

diverse_particle = dt.Sphere(
    position=lambda: np.array([32, 32]),
    radius=lambda: 500 * dt.units.nm * (0.5 + 1.5*np.random.rand()),
    position_unit="pixel",
    refractive_index=1.45 + 0.02j,
)
diverse_illuminated_sample = brightfield_microscope(diverse_particle)
diverse_clean_particle = (diverse_illuminated_sample >> dt.NormalizeMinMax()
                          >> dt.MoveAxis(2, 0)
                          >> dt.pytorch.ToTensor(dtype=torch.float))
diverse_noisy_particle = (diverse_illuminated_sample >> noise >> dt.NormalizeMinMax()
                          >> dt.MoveAxis(2, 0)
                          >> dt.pytorch.ToTensor(dtype=torch.float))
diverse_pip = diverse_noisy_particle & diverse_clean_particle

#for i in range(5):
    #input, target = diverse_pip.update().resolve()
    #plot_image(f"Input Image {i}", input.permute(1, 2, 0))
    #plot_image(f"Target Image {i}", target.permute(1, 2, 0))

#%%
class SimulatedDataset(torch.utils.data.Dataset):
    """Simulated dataset simulating pairs of noisy and clean images."""

    def __init__(self, pip, buffer_size, replace=0):
        """Initialize the dataset."""
        self.pip, self.replace = pip, replace
        self.images = [pip.update().resolve() for _ in range(buffer_size)]

    def __len__(self):
        """Return the size of the image buffer."""
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieve a noisy-clean image pair from the dataset."""
        if np.random.rand() < self.replace:
            self.images[idx] = self.pip.update().resolve()
        image_pair = self.images[idx]
        noisy_image, clean_image = image_pair[0], image_pair[1]
        return noisy_image, clean_image

diverse_dataset = SimulatedDataset(diverse_pip, buffer_size=256, replace=0.1)
diverse_loader = torch.utils.data.DataLoader(diverse_dataset, batch_size=8,
                                             shuffle=True)
#%% New class based on VAE. This uses MLP as encoder, in the latent space projection and decoder.
from typing import Optional, Sequence, Callable, List

from deeplay.components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam


import torch
import torch.nn as nn


class VariationalAutoEncoderMLP(Application):
    def __init__(
        self,
        input_size: Optional[Sequence[int]] = (64, 64),
        hidden_dim: Optional[int] = 256,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        reconstruction_loss: Optional[Callable] = nn.L1Loss(),
        latent_dim: int = 2,
        beta: float = 1,
        optimizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.flattened_size = input_size[0] * input_size[1]
        
        # Create encoder if not provided
        self.encoder = encoder or nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        
        # Create decoder if not provided
        self.decoder = decoder or nn.Sequential(
            nn.Linear(hidden_dim, self.flattened_size),
            nn.Sigmoid() if input_size == (28, 28) else nn.Identity(),  # If data MNIST
            nn.Unflatten(1, self.input_size)
        )
        
        self.reconstruction_loss = reconstruction_loss
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.optimizer = optimizer or Adam(lr=1e-3)
        
        @self.optimizer.params
        def params(self):
            return self.parameters()

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decode(z)
        return y_hat, mu, log_var
    
    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat, mu, log_var = self(x)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + self.beta * KLD
        loss = {"rec_loss": rec_loss, "KL": KLD, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss

    def compute_loss(self, y_hat, y, mu, log_var):
        rec_loss = self.reconstruction_loss(y_hat, y)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec_loss, KLD

#%% Create the VAE with latent dim 2

vae = VariationalAutoEncoderMLP(
    input_size=(64, 64),
    hidden_dim=256,
    latent_dim=2,
    reconstruction_loss=nn.L1Loss(),
    beta=1
).create()

# %%
vae_trainer = dl.Trainer(max_epochs=20, accelerator="auto")
vae_trainer.fit(vae, diverse_loader)
# %% Plot the images
for i in range(5):
    input, target = diverse_pip.update().resolve()
    
    predicted, mu, log_var = vae(input.unsqueeze(0))
    predicted = predicted.detach()
    
    input_img = input[0, :, :] 
    
    target_img = target[0, :, :] 
    
    predicted_img = predicted[0, :, :] 
    
    plot_image(f"Input Image {i}", input_img)
    plot_image(f"Target Image {i}", target_img)
    plot_image(f"Predicted Image {i}", predicted_img)
# %% b)
if not os.path.exists("MNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/MNIST_dataset")

data_dir = os.path.join("MNIST_dataset", "mnist")
train_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "train"))
test_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "test"))
files = dt.sources.Join(train_files, test_files)

print(f"Number of train images: {len(train_files)}")
print(f"Number of test images: {len(test_files)}")

image_pip = (dt.LoadImage(files.path) >> dt.NormalizeMinMax()
             >> dt.MoveAxis(2, 0) >> dt.pytorch.ToTensor(dtype=torch.float))

train_dataset = dt.pytorch.Dataset(image_pip & image_pip, inputs=train_files)
train_loader = dl.DataLoader(train_dataset, batch_size=64, shuffle=True)

#%% Create VAE with loss 
vae = VariationalAutoEncoderMLP(
    input_size=(28, 28),
    hidden_dim=512,
    latent_dim=2,
    reconstruction_loss=nn.L1Loss(),
    beta=1
).create()

# %%
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
# %%
