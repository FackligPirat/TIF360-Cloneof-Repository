#%%
import os
import deeptrack as dt
import torch
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import torch.nn as nn

if not os.path.exists("FashionMNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/FashionMNIST_dataset")

data_dir = "FashionMNIST_dataset"
train_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "train"))
test_files = dt.sources.ImageFolder(root=os.path.join(data_dir, "test"))
files = dt.sources.Join(train_files, test_files)

print(f"Number of train images: {len(train_files)}")
print(f"Number of test images: {len(test_files)}")

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]

image_pip = (dt.LoadImage(files.path) >> dt.NormalizeMinMax()
             >> dt.MoveAxis(2, 0) >> dt.pytorch.ToTensor(dtype=torch.float))
label_pip = dt.Value(files.label_name[0]) >> int

#%% Plot
fig, axs = plt.subplots(3, 10, figsize=((10, 4)))
for ax, train_file in zip(axs.ravel(),
                          np.random.choice(train_files, axs.size)):
    image, label = (image_pip & label_pip)(train_file)
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"{int(label)} {classes[int(label)]}", fontsize=9)
    ax.set_axis_off()
plt.show()
#%%

vae = dl.VariationalAutoEncoder(
    input_size=(28,28),
    channels=[32, 64], latent_dim=20,
    reconstruction_loss=torch.nn.BCELoss(reduction="mean"), beta = 1
).create()

print(vae)

def deterministic_forward(self, x):
    mu, log_var = self.encode(x)
    z = mu  # Skip sampling step
    y_hat = self.decode(z)
    return y_hat, mu, log_var

original_forward = vae.__class__.forward.__get__(vae, type(vae))

vae.forward = deterministic_forward.__get__(vae, type(vae))

train_dataset = dt.pytorch.Dataset(image_pip & image_pip, inputs=train_files)
train_loader = dl.DataLoader(train_dataset, batch_size=128, shuffle=True)
#%% Training
beta_schedule = [0.5, 1]  # KL annealing
epochs_per_stage = 10

for i, beta in enumerate(beta_schedule):
    vae.beta = beta
    if beta > 0.5:
        vae.forward = original_forward
    
    print(f"\nEpoch stage {i}, beta={beta}")

    trainer = dl.Trainer(max_epochs=epochs_per_stage, accelerator="auto")
    trainer.fit(vae, train_loader)
#%% Reconstruct
vae.eval();
vae.forward = deterministic_forward.__get__(vae, type(vae))

fig, axs = plt.subplots(2, 10, figsize=((10, 2)))
for i, test_file in enumerate(np.random.choice(test_files, 10)):
    image, label = (image_pip & label_pip)(test_file)
    axs[0, i].imshow(image.squeeze(), cmap="gray")
    axs[0, i].set_title(f"{int(label)} {classes[int(label)]}", fontsize=9)
    axs[0, i].set_axis_off()

    reconstructed_image, _,_ = vae(image.unsqueeze(0))
    axs[1, i].imshow(reconstructed_image.detach().squeeze(), cmap="gray")
    axs[1, i].set_axis_off()
plt.show()
#%% Generate new
images = vae.decode(torch.randn(30, vae.latent_dim)).detach().squeeze()

fig, axs = plt.subplots(3, 10, figsize=((10, 3)))
for ax, image in zip(axs.ravel(), images):
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()
plt.show()
#%% Latent space
steps = 6

fig, axs = plt.subplots(3, steps + 2, figsize=((10, 4)))
for i, _ in enumerate(axs):
    test_file_0, test_file_1 = np.random.choice(test_files, 2)

    image_0, label_0 = (image_pip & label_pip)(test_file_0)
    z_0 = vae.encode(image_0.unsqueeze(0))

    image_1, label_1 = (image_pip & label_pip)(test_file_1)
    z_1 = vae.encode(image_1.unsqueeze(0))

    axs[i, 0].imshow(image_0.squeeze(), cmap="gray")
    axs[i, 0].set_title(f"{int(label_0)} {classes[int(label_0)]}", fontsize=9)
    axs[i, 0].set_axis_off()

    for step in range(steps):
        z_step = z_0 + (z_1 - z_0) * step / (steps - 1)
        image_step = vae.decode(z_step).detach()
        axs[i, step + 1].imshow(image_step.squeeze(), cmap="gray")
        axs[i, step + 1].set_axis_off()
        

    axs[i, -1].imshow(image_1.squeeze(), cmap="gray")
    axs[i, -1].set_title(f"{int(label_1)} {classes[int(label_1)]}", fontsize=9)
    axs[i, -1].set_axis_off()
plt.show()