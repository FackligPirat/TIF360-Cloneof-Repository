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
from matplotlib.ticker import FixedLocator
from torch_geometric.nn.norm import LayerNorm

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
# %% Functions ex1

def create_sequences(data, input_length=24):
    X= []
    for i in range(len(data) - input_length):
        X.append(data[i:(i + input_length)])
    return np.array(X)

def add_linear_encoding(X):
    seq_len = X.shape[1]
    encoding = np.linspace(0, 1, seq_len)
    encoding = np.tile(encoding, (X.shape[0], 1))
    return np.stack([X, encoding], axis=-1)

def add_periodic_encoding(X, d_model=16):
    batch_size, seq_len = X.shape
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)

    even_dim = np.arange(0, d_model, 2)
    odd_dim = np.arange(1, d_model, 2)


    pe[:, even_dim] = np.sin(position * np.exp(-np.log(10000.0) * even_dim / d_model))
    pe[:, odd_dim] = np.cos(position * np.exp(-np.log(10000.0) * odd_dim / d_model))

    #Tile for full batch (this is what takes time)
    pe = np.tile(pe, (batch_size, 1, 1))
    X = X.reshape(batch_size, seq_len, 1)
    return np.concatenate([X, pe], axis=-1)
    
class DotProductAttention(dl.DeeplayModule):
    """Dot-product attention."""

    def __init__(self):
        """Initialize dot-product attention."""
        super().__init__()

    def forward(self, queries, keys, values):
        """Calculate dot-product attention."""
        attn_scores = (torch.matmul(queries, keys.transpose(-2, -1))
                       / (keys.size(-1) ** 0.5))
        attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_matrix, values)
        return attn_output, attn_matrix
    
def plot_attention(query_tokens, key_tokens, attn_matrix, title="Attention matrix"):
    """Plot attention."""
    fig, ax = plt.subplots()
    cax = ax.matshow(attn_matrix, cmap="Greens")
    fig.colorbar(cax)
    ax.set_title(title, fontsize=14, pad=12)
    ax.xaxis.set_major_locator(FixedLocator(range(len(key_tokens))))
    ax.yaxis.set_major_locator(FixedLocator(range(len(query_tokens))))
    ax.set_xticklabels(key_tokens, rotation=90)
    ax.set_yticklabels(query_tokens)
    plt.show()
# %% Run
temperature = data[:, 1]
input_size = 24

X = create_sequences(temperature, input_length=input_size)

encodings = {
    "No Encoding": X.reshape(X.shape[0], X.shape[1], 1),
    "Linear Encoding": add_linear_encoding(X),
    "Periodic Encoding d = 5": add_periodic_encoding(X, d_model=5),
    "Periodic Encoding d = 16": add_periodic_encoding(X, d_model=16),
    "Periodic Encoding d = 50": add_periodic_encoding(X, d_model=50),
    "Periodic Encoding d = 100": add_periodic_encoding(X, d_model=100),
}

attention = DotProductAttention()

for name, X_encoded in encodings.items():
    sample = torch.tensor(X_encoded[0:1], dtype=torch.float32)  # shape: (1, seq_len, dim)
    queries = keys = values = sample

    attn_output, attn_matrix = attention(queries, keys, values)

    tokens = [f"t-{i}" for i in range(sample.shape[1])]
    plot_attention(tokens, tokens, attn_matrix[0].detach().numpy(), title=name)
# %%
