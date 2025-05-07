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
from torch_geometric.data import Data


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
#%%
n_samples, n_features = data.shape[0], data.shape[1]
past_seq = 2 * daily_samples
lag = 72
temp_idx = 1  # Temperature (Celsius) index.

in_sequences, targets = [], []
for i in range(past_seq, n_samples - lag):
    in_sequences.append(data[i - past_seq:i, :])
    targets.append(data[i + lag:i + lag + 1, temp_idx])
in_sequences, targets = np.asarray(in_sequences), np.asarray(targets)

sources = dt.sources.Source(inputs=in_sequences, targets=targets)
train_sources, val_sources = dt.sources.random_split(sources, [0.8, 0.2])

train_mean = np.mean([src["inputs"] for src in train_sources], axis=(0, 1))
train_std = np.std([src["inputs"] for src in train_sources], axis=(0, 1))

inputs_pipeline = (dt.Value(sources.inputs - train_mean) / train_std
                   >> dt.pytorch.ToTensor(dtype=torch.float))
targets_pipeline = (dt.Value(sources.targets - train_mean[temp_idx]) 
                    / train_std[temp_idx])

train_dataset = dt.pytorch.Dataset(inputs_pipeline & targets_pipeline,
                                   inputs=train_sources)
val_dataset = dt.pytorch.Dataset(inputs_pipeline & targets_pipeline,
                                 inputs=val_sources)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

temperature = data[:, temp_idx]
benchmark_celsius = np.mean(
    np.abs(
        temperature[daily_samples + lag::daily_samples]
        - temperature[lag:-(daily_samples - lag):daily_samples]
    )
)
benchmark = benchmark_celsius / train_std[temp_idx]

def plot_training(epochs, train_losses, val_losses, benchmark):
    """Plot the training and validation losses."""
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), val_losses, "--", label="Validation Loss")
    plt.plot([0, epochs - 1], [benchmark, benchmark], ":k", label="Benchmark")
    plt.xlabel("Epoch")
    plt.xlim([0, epochs - 1])
    plt.ylabel("Loss")
    plt.ylim(top = 0.7)
    plt.legend()
    plt.show()

def collate(batch_of_samples):
    sequences, labels, batch_indices = [], [], []
    for batch_index, (input_seq, target) in enumerate(batch_of_samples):
        sequence = torch.tensor(input_seq) #Changed to work with Jenna
        sequences.append(sequence)
        batch_indices.append(torch.ones(len(sequence), dtype=torch.long) * batch_index)
        labels.append(torch.tensor(target))
    return Data(sequences=torch.cat(sequences),
                batch_indices=torch.cat(batch_indices),
                y=torch.stack(labels).float())

train_dataloader = \
    DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
val_dataloader = \
    DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate)

class MultiHeadAttentionLayer(dl.DeeplayModule):
    """Multi-head attention layer with masking."""

    def __init__(self, num_features, num_heads):
        """Initialize multi-head attention."""
        super().__init__()
        self.num_features, self.num_heads = num_features, num_heads
        self.head_dim = num_features // num_heads  # Must be integer.

        self.Wq = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wk = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wv = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wout = dl.Layer(torch.nn.Linear, num_features, num_features)

    def forward(self, in_sequence, batch_indices):
        """Apply the multi-head attention mechanism to the input sequence."""
        seq_len, embed_dim = in_sequence.shape
        Q = self.Wq(in_sequence)
        Q = Q.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.Wk(in_sequence)
        K = K.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.Wv(in_sequence)
        V = V.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

        attn_scores = (torch.matmul(Q, K.transpose(-2, -1))
                       / (self.head_dim ** 0.5))

        attn_mask = torch.eq(batch_indices.unsqueeze(1),
                             batch_indices.unsqueeze(0))
        attn_mask = attn_mask.unsqueeze(0)
        attn_scores = attn_scores.masked_fill(attn_mask == False,
                                              float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        attn_output = attn_output.view(seq_len, self.num_features)
        return self.Wout(attn_output)
    
class TransformerEncoderLayer(dl.DeeplayModule):
    """Transformer encoder layer."""

    def __init__(self, num_features, num_heads, feedforward_dim, dropout=0.0):
        """Initialize transformer encoder layer."""
        super().__init__()

        self.self_attn = MultiHeadAttentionLayer(num_features, num_heads)
        self.attn_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.attn_skip = dl.Add()
        self.attn_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

        self.feedforward = dl.Sequential(
            dl.Layer(torch.nn.Linear, num_features, feedforward_dim),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, feedforward_dim, num_features),
        )
        self.feedforward_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.feedforward_skip = dl.Add()
        self.feedforward_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

    def forward(self, in_sequence, batch_indices):
        """Refine sequence via attention and feedforward layers."""
        attns = self.self_attn(in_sequence, batch_indices)
        attns = self.attn_dropout(attns)
        attns = self.attn_skip(in_sequence, attns)
        attns = self.attn_norm(attns, batch_indices)

        out_sequence = self.feedforward(attns)
        out_sequence = self.feedforward_dropout(out_sequence)
        out_sequence = self.feedforward_skip(attns, out_sequence)
        out_sequence = self.feedforward_norm(out_sequence, batch_indices)

        return out_sequence
    
class TransformerEncoderModel(dl.DeeplayModule):
    """Transformer encoder model."""

    def __init__(self, num_features, num_heads, feedforward_dim,
                 num_layers, out_dim, dropout=0.0, input_dim=in_sequences.shape[-1]):
        """Initialize transformer encoder model."""
        super().__init__()
        self.num_features = num_features

        self.input_projection = dl.Layer(torch.nn.Linear, input_dim, num_features)

        self.pos_encoder = dl.IndexedPositionalEmbedding(num_features)
        self.pos_encoder.dropout.configure(p=dropout)

        self.transformer_block = dl.LayerList()
        for _ in range(num_layers):
            self.transformer_block.append(TransformerEncoderLayer(
                    num_features, num_heads, feedforward_dim, dropout=dropout,
            ))

        self.out_block = dl.Sequential(
            dl.Layer(torch.nn.Dropout, dropout),
            dl.Layer(torch.nn.Linear, num_features, num_features // 2),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, num_features // 2, out_dim),
            #dl.Layer(torch.nn.Sigmoid),
        )

    def forward(self, dict):
        """Predict sentiment of movie reviews."""
        in_sequence, batch_indices = dict["sequences"], dict["batch_indices"]

        embeddings = self.input_projection(in_sequence) * self.num_features ** 0.5
        pos_embeddings = self.pos_encoder(embeddings, batch_indices)

        out_sequence = pos_embeddings
        for transformer_layer in self.transformer_block:
            out_sequence = transformer_layer(out_sequence, batch_indices)

        batch_size = torch.max(batch_indices) + 1
        aggregates = torch.zeros(batch_size, self.num_features,
                                 device=out_sequence.device)
        for batch_index in torch.unique(batch_indices):
            mask = batch_indices == batch_index
            aggregates[batch_index] = out_sequence[mask].mean(dim=0)

        pred_sentiment = self.out_block(aggregates).squeeze()
        return pred_sentiment
#%%
epochs = 100

model = TransformerEncoderModel(
    input_dim=in_sequences.shape[-1],
    num_features=300,
    num_heads=12,
    feedforward_dim=512,
    num_layers=4,
    out_dim=1,
    dropout=0.1,
).create()

print(model)

regressor = dl.Regressor(
    model=model, optimizer=dl.Adam(lr=0.001),
).create()

trainer = dl.Trainer(max_epochs=epochs, accelerator="gpu")  ###
trainer.fit(regressor, train_dataloader, val_dataloader)

train_losses = trainer.history.history["train_loss_epoch"]["value"]
val_losses = trainer.history.history["val_loss_epoch"]["value"][1:]
plot_training(epochs, train_losses, val_losses, benchmark)
# %%
