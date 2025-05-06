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
#%% Functions
def create_sequences(data, input_length=24):
    X = []
    for i in range(len(data) - input_length):
        X.append(data[i:(i + input_length)])
    return np.array(X)

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

class Time2Vec(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.omega = nn.Parameter(torch.randn(d_model))
        self.tau = nn.Parameter(torch.randn(d_model))

    def forward(self, batch_size):
        t = torch.arange(self.seq_len).float().unsqueeze(1)
        omega = self.omega.unsqueeze(0)
        tau = self.tau.unsqueeze(0) 

        #
        angles = t @ omega + tau
        time2vec = torch.zeros_like(angles)
        time2vec[:, 0] = angles[:, 0]
        time2vec[:, 1:] = torch.sin(angles[:, 1:])

        time2vec = time2vec.unsqueeze(0).repeat(batch_size, 1, 1)
        return time2vec
#%% Run
temperature = data[:, 1]

input_length = 24

X = create_sequences(temperature, input_length= input_length)

torch.manual_seed(12)
np.random.seed(12)


batch_size = X.shape[0]

X_input = X[:, :input_length].reshape(batch_size, input_length, 1)

time2vec = Time2Vec(seq_len=input_length, d_model=16)
encoding = time2vec(batch_size)  

X_t2v = torch.cat([torch.tensor(X_input, dtype=torch.float32), encoding], dim=-1)

attention = DotProductAttention()
sample = X_t2v[0:1] 
queries = keys = values = sample

attn_output, attn_matrix = attention(queries, keys, values)

tokens = [f"t-{i}" for i in range(sample.shape[1])]
plot_attention(tokens, tokens, attn_matrix[0].detach().numpy(), title="Time2Vec Encoding")
# %%
