#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import deeptrack as dt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import deeplay as dl

dataframe = pd.read_csv("jena_climate_2009_2016.csv", index_col=0)
data = dataframe.values
header = dataframe.columns.tolist()

start, days, daily_samples = 0, 14, 144
end = start + daily_samples * days

fig, axs = plt.subplots(7, 2, figsize=(16, 12), sharex=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(np.arange(start, end), data[start:end, i], label=header[i])
    ax.set_xlim(start, end)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=20)

    for day in range(1, days):
        ax.axvline(x=start + daily_samples * day,
                   color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
# %%
df = pd.read_csv('jena_climate_2009_2016.csv')
temperature = df['T (degC)'].values

def create_sequences(data, input_length=144, output_length=24):
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i:(i + input_length)])
        y.append(data[(i + input_length):(i + input_length + output_length)])
    return np.array(X), np.array(y)

X, y = create_sequences(temperature)

def add_linear_encoding(X):
    seq_len = X.shape[1]
    encoding = np.linspace(0, 1, seq_len)
    encoding = np.tile(encoding, (X.shape[0], 1))
    return np.stack([X, encoding], axis=-1)  # New input shape: (batch, seq, 2)

def add_periodic_encoding(X, d_model=16):
    batch_size, seq_len = X.shape
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = np.tile(pe, (batch_size, 1, 1))
    X = X.reshape(batch_size, seq_len, 1)
    return np.concatenate([X, pe], axis=-1)  # (batch, seq, d_model+1)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, seq, input_dim)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq, batch, embed_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.output_proj(x).squeeze(-1)
    
def plot_attention(attention_matrix):
    plt.imshow(attention_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()