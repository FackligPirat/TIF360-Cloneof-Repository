#%%
import os
import deeptrack as dt
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex
import deeplay as dl
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping

if not os.path.exists("tissue_images_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/tissue_images_dataset")

raw_path = os.path.join("tissue_images_dataset", "stack1", "raw")
seg_path = os.path.join("tissue_images_dataset", "stack1", "labels")

raw_paths = dt.sources.ImageFolder(root=raw_path)
seg_paths = dt.sources.ImageFolder(root=seg_path)

paths = dt.sources.Source(raw=raw_paths, label=seg_paths)
train_paths, val_paths, test_paths = \
    dt.sources.random_split(paths, [0.8, 0.1, 0.1])

train_srcs = train_paths.product(flip_ud=[True, False], flip_lr=[True, False])
val_srcs = val_paths.constants(flip_ud=False, flip_lr=False)
test_srcs = test_paths.constants(flip_ud=False, flip_lr=False)

sources = dt.sources.Join(train_srcs, val_srcs, test_srcs)

print(f"Raw images: {len(raw_paths)}")
print(f"Segmentation labels: {len(seg_paths)}")

#%% Define preprocessing pipeline and datasets
def select_labels(class_labels):
    """Create a function to filter and remap labels in a segmentation map."""
    def inner(segmentation):
        seg = segmentation.copy()
        mask = seg * np.isin(seg, class_labels).astype(np.uint8)
        new_seg = (np.select([mask == c for c in class_labels],
                             np.arange(len(class_labels)) + 1)
                   .astype(np.uint8).squeeze())
        one_hot_encoded_seg = np.eye(len(class_labels) + 1)[new_seg]
        return one_hot_encoded_seg        
    return inner

im_pip = dt.LoadImage(sources.raw.path) >> dt.NormalizeMinMax()
seg_pip = (dt.LoadImage(sources.label.path)
           >> dt.Lambda(select_labels, class_labels=[255, 191]))
pip = ((im_pip & seg_pip) >> dt.FlipLR(sources.flip_lr)
       >> dt.FlipUD(sources.flip_ud) >> dt.MoveAxis(2, 0)
       >> dt.pytorch.ToTensor(dtype=torch.float))

train_dataset = dt.pytorch.Dataset(pip, train_srcs)
val_dataset = dt.pytorch.Dataset(pip, val_srcs)
test_dataset = dt.pytorch.Dataset(pip, test_srcs)

#%% Visualization
image, segmentation = train_dataset[0]

fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].imshow(image.permute(1, 2, 0), cmap="gray")
axs[0].set_title("Image", fontsize=16)
axs[0].set_axis_off()

for i in range(segmentation.shape[0]):
    segmentation_color = torch.ones_like(segmentation)
    for j in range(segmentation.shape[0]):
        if j != i: segmentation_color[j, ...] = 1 - segmentation[i, ...]
    axs[i + 1].imshow(segmentation_color.permute(1, 2, 0))
    axs[i + 1].set_title(f"Ground Truth Ch. {i}", fontsize=16)
    axs[i + 1].set_axis_off()

plt.tight_layout()
plt.show()

#%% Metric definition
class ArgmaxJI(MulticlassJaccardIndex):
    """Compute Jaccard Index for multi-class predictions after argmax."""

    def update(self, preds, targets):
        """Update Jaccard Index using argmax of class predictions."""
        super().update(preds.argmax(dim=1), targets.argmax(dim=1))

#%% NoSkip module
class NoSkip(dl.DeeplayModule):
    def forward(self, encoder_output, decoder_input):
        return decoder_input

ji_metric = ArgmaxJI(num_classes=3)

#%% UNet definitions
channels = [16, 32, 64, 128]

# With skip connections
unet_with_skip = dl.UNet2d(
    in_channels=1, channels=channels, out_channels=3, skip=dl.Cat()
)
reg_with_skip = dl.Regressor(
    model=unet_with_skip,
    loss=torch.nn.CrossEntropyLoss(),
    metrics=[ji_metric],
    optimizer=dl.Adam(),
).create()

# Without skip connections
unet_no_skip = dl.UNet2d(
    in_channels=1, channels=channels, out_channels=3, skip=NoSkip()
)
reg_no_skip = dl.Regressor(
    model=unet_no_skip,
    loss=torch.nn.CrossEntropyLoss(),
    metrics=[ji_metric],
    optimizer=dl.Adam(),
).create()

#%% DataLoaders
train_loader = dl.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = dl.DataLoader(val_dataset, batch_size=2)

#%% ADD training without early stopping
#%% Train with skip connections
early_stop = EarlyStopping(monitor="valArgmaxJI_epoch", mode="max", patience=5)
logger_skip = CSVLogger("logs", name="with_skip")
trainer_skip = dl.Trainer(max_epochs=100, logger=logger_skip, callbacks=[early_stop])
trainer_skip.fit(reg_with_skip, train_loader, val_loader)

#%% Train without skip connections
ji_metric = ArgmaxJI(num_classes=3)  # reset metric instance
reg_no_skip = dl.Regressor(
    model=unet_no_skip,
    loss=torch.nn.CrossEntropyLoss(),
    metrics=[ji_metric],
    optimizer=dl.Adam(),
).create()

logger_noskip = CSVLogger("logs", name="without_skip")
trainer_noskip = dl.Trainer(max_epochs=100, logger=logger_noskip, callbacks=[early_stop])
trainer_noskip.fit(reg_no_skip, train_loader, val_loader)

#%% Plot training metrics for both
def plot_training_metrics(metrics_skip, metrics_noskip):
    fig, axs = plt.subplots(2, figsize=(6, 4))

    axs[0].plot(metrics_skip["epoch"], metrics_skip["train_loss_epoch"], label="With Skip")
    axs[0].plot(metrics_skip["epoch"], metrics_skip["val_loss_epoch"], label="Val With Skip")
    axs[0].plot(metrics_noskip["epoch"], metrics_noskip["train_loss_epoch"], label="No Skip")
    axs[0].plot(metrics_noskip["epoch"], metrics_noskip["val_loss_epoch"], label="Val No Skip")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(metrics_skip["epoch"], metrics_skip["trainArgmaxJI_epoch"], label="Train With Skip")
    axs[1].plot(metrics_skip["epoch"], metrics_skip["valArgmaxJI_epoch"], label="Val With Skip")
    axs[1].plot(metrics_noskip["epoch"], metrics_noskip["trainArgmaxJI_epoch"], label="Train No Skip")
    axs[1].plot(metrics_noskip["epoch"], metrics_noskip["valArgmaxJI_epoch"], label="Val No Skip")
    axs[1].set_ylabel("Jaccard Index")
    axs[1].legend()

    axs[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()

metrics_skip = pd.read_csv(os.path.join(logger_skip.log_dir, "metrics.csv")).ffill()
metrics_noskip = pd.read_csv(os.path.join(logger_noskip.log_dir, "metrics.csv")).ffill()
plot_training_metrics(metrics_skip, metrics_noskip)

#%% Test prediction
test_loader = dl.DataLoader(test_dataset, batch_size=2, shuffle=False)
pred_skip = torch.cat(trainer_skip.predict(reg_with_skip, test_loader), dim=0)
pred_noskip = torch.cat(trainer_noskip.predict(reg_no_skip, test_loader), dim=0)

#%% Visual comparison
test_image, test_seg = test_dataset[0]

fig, axs = plt.subplots(1, 4, figsize=(16, 6))
axs[0].imshow(test_image[0], cmap="gray")
axs[0].set_title("Image", fontsize=16)
axs[0].set_axis_off()

axs[1].imshow(test_seg.argmax(dim=0))
axs[1].set_title("Ground Truth", fontsize=16)
axs[1].set_axis_off()

axs[2].imshow(pred_skip[0].argmax(dim=0))
axs[2].set_title("Prediction (With Skip)", fontsize=16)
axs[2].set_axis_off()

axs[3].imshow(pred_noskip[0].argmax(dim=0))
axs[3].set_title("Prediction (No Skip)", fontsize=16)
axs[3].set_axis_off()

plt.tight_layout()
plt.show()

#%% Test Jaccard Index
ji_metric = ArgmaxJI(num_classes=3)
ji_with_skip = ji_metric(pred_skip[0].unsqueeze(0), test_seg.unsqueeze(0)).item()
ji_metric.reset()
ji_without_skip = ji_metric(pred_noskip[0].unsqueeze(0), test_seg.unsqueeze(0)).item()

print(f"Test Jaccard Index (With Skip): {ji_with_skip:.4f}")
print(f"Test Jaccard Index (No Skip): {ji_without_skip:.4f}")
