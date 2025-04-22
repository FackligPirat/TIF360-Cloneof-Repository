import os
from torchvision.datasets.utils import _extract_zip, download_url

dataset_path = "cell_detection_dataset"
if not os.path.exists(dataset_path):
    url = ("http://data.celltrackingchallenge.net/training-datasets/"
           "BF-C2DL-HSC.zip")
    download_url(url, ".")
    _extract_zip("BF-C2DL-HSC.zip", dataset_path, None)
    os.remove("BF-C2DL-HSC.zip")

dir = os.path.join(dataset_path, "BF-C2DL-HSC")

import glob
import deeptrack as dt
from skimage.measure import regionprops

sources = dt.sources.Source(
    image_path=sorted(glob.glob(os.path.join(dir, "02", "*.tif"))),
    label_path=sorted(glob.glob(os.path.join(dir, "02_GT", "TRA", "*.tif"))),
)

image_pip = dt.LoadImage(sources.image_path)[300:850, :300] / 256
props_pip = dt.LoadImage(sources.label_path)[300:850, :300] >> regionprops

pip = image_pip & props_pip

import matplotlib.pyplot as plt
import imagecodecs

plt.figure(figsize=(15, 10))

for plt_index, data_index in enumerate([0, 300, 600, 900, 1200, 1500]):
    image, *props = pip(sources[data_index])

    plt.subplot(1, 6, plt_index + 1)
    plt.imshow(image, cmap="gray")
    for prop in props:
        plt.scatter(prop.centroid[1], prop.centroid[0], s=5, color="red")
    plt.axis("off")
plt.tight_layout()
plt.show()

crop_frame_index, crop_size = 282, 50
crop_x0, crop_y0 = 295 - crop_size // 2, 115 - crop_size // 2

image, *props = pip(sources[crop_frame_index])
crop = image[crop_x0:crop_x0 + crop_size, crop_y0:crop_y0 + crop_size]

from matplotlib.patches import Rectangle

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.gca().add_patch(Rectangle((crop_y0, crop_x0), crop_size, crop_size,
                              linewidth=1, edgecolor="r", facecolor="none"))

plt.subplot(1, 2, 2)
plt.imshow(crop, cmap="gray")

plt.show()

import numpy as np
import torch

print(np.shape(crop))
train_pip = (dt.Value(crop)
             >> dt.Multiply(lambda: np.random.uniform(0.9, 1.1))
             >> dt.Add(lambda: np.random.uniform(-0.1, 0.1))
             >> dt.MoveAxis(-1, 0) >> dt.pytorch.ToTensor(dtype=torch.float32))

import deeplay as dl

train_dataset = dt.pytorch.Dataset(train_pip, length=400, replace=False)
dataloader = dl.DataLoader(train_dataset, batch_size=8, shuffle=True)

lodestar = dl.LodeSTAR(n_transforms=4, optimizer=dl.Adam(lr=1e-4)).build()
trainer = dl.Trainer(max_epochs=200)
trainer.fit(lodestar, dataloader)
#%% Plot prediction
image_index = 1500
image, *props = pip(sources[image_index])
torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
prediction = lodestar(torch_image)[0].detach().numpy()
x, y, rho = prediction[0], prediction[1], prediction[-1]

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(rho, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image, cmap="gray")
plt.scatter(y.flatten(), x.flatten(), alpha=rho.flatten() / rho.max(), s=5)
plt.axis("off")
plt.xlim(0, 299)
plt.ylim(0, 549)
plt.gca().invert_yaxis()

plt.show()