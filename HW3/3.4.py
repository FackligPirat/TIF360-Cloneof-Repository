#%% Import libraries and define some functions
import contractions, re, spacy, unicodedata
from collections import Counter
import numpy as np
import deeplay as dl
import deeptrack as dt
import torch
import os
from torchmetrics.text import BLEUScore
from torchvision.datasets.utils import download_url, extract_archive

tokenizers = {"eng": spacy.blank("en"), "swe": spacy.blank("sv")}

#regular_expression = r"^[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ.,!?¡¿/:()]+$" # Old spanish
regular_expression = r"^[a-zA-Z0-9åäöÅÄÖ.,!?¡¿/:()]+$"

pattern = re.compile(unicodedata.normalize("NFC", regular_expression))

def tokenize(text, lang="eng"):
    """Tokenize text."""
    swaps = {"’": "'", "‘": "'", "“": '"', "”": '"', "´": "'", "´´": '"'}
    for old, new in swaps.items():
        text = text.replace(old, new)
    text = contractions.fix(text) if lang == "eng" else text
    tokens = tokenizers[lang](text)
    return [token.text for token in tokens if pattern.match(token.text)]

def corpus_iterator(filename, lang, lang_position):
    """Read and tokenize texts by iterating through a corpus file."""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            sentences = line.strip().split("\t")
            sentence = unicodedata.normalize("NFC", sentences[lang_position])
            yield tokenize(sentence, lang)

for tokens_eng, tokens_spa in zip(
    corpus_iterator(filename="swe.txt", lang="eng", lang_position=0),
    corpus_iterator(filename="swe.txt", lang="swe", lang_position=1),
    ):
    print(f"{tokens_eng} {tokens_spa}")

class Vocab:
    """Vocabulary as callable dictionary."""

    def __init__(self, vocab_dict, unk_token="<unk>"):
        """Initialize vocabulary."""
        self.vocab_dict, self.unk_token = vocab_dict, unk_token
        self.default_index = vocab_dict.get(unk_token, -1)
        self.index_to_token = {idx: token for token, idx in vocab_dict.items()}

    def __call__(self, token_or_tokens):
        """Return the index(es) for given token or list of tokens."""
        if not isinstance(token_or_tokens, list):
            return self.vocab_dict.get(token_or_tokens, self.default_index)
        else:
            return [self.vocab_dict.get(token, self.default_index)
                    for token in token_or_tokens]

    def set_default_index(self, index):
        """Set default index for unknown tokens."""
        self.default_index = index

    def lookup_token(self, index_or_indices):
        """Retrieve token corresponding to given index or list of indices."""
        if not isinstance(index_or_indices, list):
            return self.index_to_token.get(int(index_or_indices),
                                           self.unk_token)
        else:
            return [self.index_to_token.get(int(index), self.unk_token)
                    for index in index_or_indices]

    def get_tokens(self):
        """Return a list of tokens ordered by their index."""
        tokens = [None] * len(self.index_to_token)
        for index, token in self.index_to_token.items():
            tokens[index] = token
        return tokens

    def __iter__(self):
        """Iterate over the tokens in the vocabulary."""
        return iter(self.vocab_dict)

    def __len__(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.vocab_dict)

    def __contains__(self, token):
        """Check if a token is in the vocabulary."""
        return token in self.vocab_dict

vocab_dict = {"hello": 0, "world": 1, "<unk>": 2}
vocab = Vocab(vocab_dict)

vocab.lookup_token(5)

def build_vocab_from_iterator(iterator, specials=None, min_freq=1):
    """Build vocabulary from an iterator over tokenized sentences."""
    token_freq = Counter(token for tokens in iterator for token in tokens)
    vocab, index = {}, 0
    if specials:
        for token in specials:
            vocab[token] = index
            index += 1
    for token, freq in token_freq.items():
        if freq >= min_freq:
            vocab[token] = index
            index += 1
    return vocab

def build_vocab(filename, lang, lang_position, specials=["<unk>"], min_freq=5):
    """Build vocabulary."""
    vocab_dict = build_vocab_from_iterator(
        corpus_iterator(filename, lang, lang_position), specials, min_freq,
    )
    vocab = Vocab(vocab_dict, unk_token=specials[0])
    vocab.set_default_index(vocab(specials[0]))
    return vocab

in_lang, out_lang, filename = "eng", "swe", "swe.txt"
specials = ["<pad>", "<sos>", "<eos>", "<unk>"]

in_vocab = build_vocab(filename, in_lang, lang_position=0, specials=specials)
out_vocab = build_vocab(filename, out_lang, lang_position=1, specials=specials)

def all_words_in_vocab(sentence, vocab):
    """Check whether all words in a sentence are present in a vocabulary."""
    return all(word in vocab for word in sentence)

def pad(tokens, max_length=10):
    """Pad sequence of tokens."""
    padding_length = max_length - len(tokens)
    return ["<sos>"] + tokens + ["<eos>"] + ["<pad>"] * padding_length

def process(filename, in_lang, out_lang, in_vocab, out_vocab, max_length=10):
    """Process language corpus."""
    in_sequences, out_sequences = [], []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            sentences = line.strip().split("\t")
            in_tokens = tokenize(unicodedata.normalize("NFC", sentences[0]),
                                 in_lang)
            out_tokens = tokenize(unicodedata.normalize("NFC", sentences[1]),
                                  out_lang)

            if (all_words_in_vocab(in_tokens, in_vocab)
                and len(in_tokens) <= max_length
                and all_words_in_vocab(out_tokens, out_vocab)
                and len(out_tokens) <= max_length):

                padded_in_tokens = pad(in_tokens)
                in_sequence = in_vocab(padded_in_tokens)
                in_sequences.append(in_sequence)

                padded_out_tokens = pad(out_tokens)
                out_sequence = out_vocab(padded_out_tokens)
                out_sequences.append(out_sequence)
    return np.array(in_sequences), np.array(out_sequences)

in_sequences, out_sequences = \
    process(filename, in_lang, out_lang, in_vocab, out_vocab)

sources = dt.sources.Source(inputs=in_sequences, targets=out_sequences)
train_sources, test_sources = dt.sources.random_split(sources, [0.85, 0.15])

inputs_pip = dt.Value(sources.inputs) >> dt.pytorch.ToTensor(dtype=torch.int)
outputs_pip = dt.Value(sources.targets) >> dt.pytorch.ToTensor(dtype=torch.int)

train_dataset = \
    dt.pytorch.Dataset(inputs_pip & outputs_pip, inputs=train_sources)
test_dataset = \
    dt.pytorch.Dataset(inputs_pip & outputs_pip, inputs=test_sources)

train_loader = dl.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = dl.DataLoader(test_dataset, batch_size=256, shuffle=False)

class Seq2SeqEncoder(dl.DeeplayModule):
    """Sequence-to-sequence encoder."""

    def __init__(self, vocab_size, in_feats=300, hidden_feats=128,
                 hidden_layers=1, dropout=0.0):
        """Initialize sequence-to-sequence encoder."""
        super().__init__()
        self.hidden_feats, self.hidden_layers = hidden_feats, hidden_layers

        self.embedding = dl.Layer(torch.nn.Embedding, vocab_size, in_feats)
        self.rnn = dl.Layer(torch.nn.GRU, input_size=in_feats,
                            hidden_size=hidden_feats, num_layers=hidden_layers,
                            dropout=(0 if hidden_layers == 1 else dropout),
                            bidirectional=True, batch_first=True)

    def forward(self, in_sequences, contexts=None):
        """Calculate the encoded sequences and contexts."""
        in_embeddings = self.embedding(in_sequences)
        encoded_sequences, contexts = self.rnn(in_embeddings, contexts)
        encoded_sequences = (encoded_sequences[:, :, :self.hidden_feats]
                             + encoded_sequences[:, :, self.hidden_feats:])
        contexts = contexts[:self.hidden_layers]
        return encoded_sequences, contexts

class Seq2SeqDecoder(dl.DeeplayModule):
    """Sequence-to-sequence decoder."""

    def __init__(self, vocab_size, in_feats=300, hidden_feats=128,
                 hidden_layers=1, dropout=0.0):
        """Initialize sequence-to-sequence decoder."""
        super().__init__()

        self.embedding = dl.Layer(torch.nn.Embedding, vocab_size, in_feats)
        self.rnn = dl.Layer(torch.nn.GRU, input_size=in_feats,
                            hidden_size=hidden_feats, num_layers=hidden_layers,
                            bidirectional=False, batch_first=True,
                            dropout=(0 if hidden_layers == 1 else dropout))
        self.dense = dl.Layer(torch.nn.Linear, hidden_feats, vocab_size)
        self.softmax = dl.Layer(torch.nn.Softmax, dim=-1)

    def forward(self, decoder_in_values, contexts):
        """Calculate the decoder outputs and contexts."""
        out_embeddings = self.embedding(decoder_in_values)
        decoder_outputs, contexts = self.rnn(out_embeddings, contexts)
        decoder_outputs = self.dense(decoder_outputs)
        decoder_outputs = self.softmax(decoder_outputs)
        return decoder_outputs, contexts

class Seq2SeqModel(dl.DeeplayModule):
    """Sequence-to-sequence model with evaluation method."""

    def __init__(self, in_vocab_size=None, out_vocab_size=None, embed_dim=300,
                 hidden_feats=128, hidden_layers=1, dropout=0.0,
                 teacher_prob=1.0):
        """Initialize the sequence-to-sequence model."""
        super().__init__()
        self.in_vocab_size, self.out_vocab_size = in_vocab_size, out_vocab_size
        self.teacher_prob = teacher_prob

        self.encoder = Seq2SeqEncoder(in_vocab_size, embed_dim, hidden_feats,
                                      hidden_layers, dropout)
        self.decoder = Seq2SeqDecoder(out_vocab_size, embed_dim, hidden_feats,
                                      hidden_layers, dropout)

    def forward(self, batch):
        """Calculate the decoder output vectors for the input sequences."""
        in_sequences, out_sequences = batch
        num_sequences, sequence_length = in_sequences.size()
        device = next(self.encoder.parameters()).device

        _, contexts = self.encoder(in_sequences)

        decoder_outputs_vec = torch.zeros(num_sequences, sequence_length,
                                          self.out_vocab_size).to(device)
        decoder_in_values = torch.full(size=(num_sequences, 1),
                                       fill_value=1, device=device)  # <sos>
        for t in range(sequence_length):
            decoder_outputs, contexts = \
                self.decoder(decoder_in_values, contexts)
            decoder_outputs_vec[:, t, :] = decoder_outputs.squeeze(1)

            if (np.random.rand() < self.teacher_prob
                and t < sequence_length - 1):  # Teacher forcing.
                decoder_in_values = \
                    out_sequences[:, t + 1].unsqueeze(-1).to(device)
            else:  # Model prediction.
                _, top_decoder_outputs = decoder_outputs.topk(1)
                decoder_in_values = \
                    top_decoder_outputs.squeeze(-1).detach().to(device)
        
        return decoder_outputs_vec

    def evaluate(self, in_sequences):
        """Evaluate model."""
        num_sequences, sequence_length = in_sequences.size()
        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            _, contexts = self.encoder(in_sequences)

        pred_sequences = torch.zeros(num_sequences, sequence_length).to(device)
        decoder_in_values = torch.full(size=(num_sequences, 1),
                                       fill_value=1, device=device)  # <sos>
        for t in range(sequence_length):
            with torch.no_grad():
                decoder_outputs, contexts = \
                    self.decoder(decoder_in_values, contexts)
            _, top_decoder_outputs = decoder_outputs.topk(1)
            pred_sequences[:, t] = top_decoder_outputs.squeeze()

            decoder_in_values = top_decoder_outputs.squeeze(-1).detach()

        return pred_sequences

def maskedNLL(decoder_outputs, out_sequences, padding=0):
    """Calculate the masked negative log-likelihood (NLL) loss."""
    flat_pred_sequences = decoder_outputs.view(-1, decoder_outputs.shape[-1])
    flat_target_sequences = out_sequences.view(-1, 1)
    pred_probs = torch.gather(flat_pred_sequences, 1, flat_target_sequences)

    nll = - torch.log(pred_probs)

    mask = out_sequences != padding
    masked_nll = nll.masked_select(mask.view(-1, 1))

    return masked_nll.mean()  # Loss.

class Seq2Seq(dl.Application):
    """Application for the sequence-to-sequence model."""

    def __init__(self, in_vocab, out_vocab, teacher_prob=1.0):
        """Initialize the application."""
        super().__init__(loss=maskedNLL, optimizer=dl.Adam(lr=1e-3))
        self.model = Seq2SeqModel(in_vocab_size=len(in_vocab),
                                  out_vocab_size=len(out_vocab),
                                  teacher_prob=teacher_prob)

    def train_preprocess(self, batch):
        """Adjust the target sequence by shifting it one position backward."""
        in_sequences, out_sequences = batch
        shifted_out_sequences = \
            torch.cat((out_sequences[:, 1:], out_sequences[:, -1:]), dim=1)
        return (in_sequences, out_sequences), shifted_out_sequences

    def forward(self, batch):
        """Perform forward pass."""
        return self.model(batch)
#%% Load pre-trained Embeddings

glove_folder = ".glove_cache"
if not os.path.exists(glove_folder):
    os.makedirs(glove_folder, exist_ok=True)
    url = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
    download_url(url, glove_folder)
    zip_filepath = os.path.join(glove_folder, "glove.42B.300d.zip")
    extract_archive(zip_filepath, glove_folder)
    os.remove(zip_filepath)

def load_glove_embeddings(glove_file):
    """Load GloVe embeddings."""
    glove_embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            glove_embeddings[word] = np.round(
                np.asarray(values[1:], dtype="float32"), decimals=6,
            )
    return glove_embeddings

def get_glove_embeddings(vocab, glove_embeddings, embed_dim):
    """Get GloVe embeddings for a vocabulary."""
    embeddings = torch.zeros((len(vocab), embed_dim), dtype=torch.float32)
    for i, token in enumerate(vocab):
        embedding = glove_embeddings.get(token)
        if embedding is None:
            embedding = glove_embeddings.get(token.lower())
        if embedding is not None:
            embeddings[i] = torch.tensor(embedding, dtype=torch.float32)
    return embeddings

glove_file = os.path.join(glove_folder, "glove.42B.300d.txt")
glove_embeddings, glove_dim = load_glove_embeddings(glove_file), 300

embeddings_in = get_glove_embeddings(in_vocab.get_tokens(),
                                     glove_embeddings, glove_dim)
embeddings_out = get_glove_embeddings(out_vocab.get_tokens(),
                                      glove_embeddings, glove_dim)

num_specials = len(specials)
embeddings_in[1:num_specials] = torch.rand(num_specials - 1, glove_dim) * 0.01
embeddings_out[1:num_specials] = torch.rand(num_specials - 1, glove_dim) * 0.01

#%% Training Seq-to-Seq

seq2seq = Seq2Seq(in_vocab=in_vocab, out_vocab=out_vocab, teacher_prob=0.85)
seq2seq = seq2seq.create()

seq2seq.model.encoder.embedding.weight.data = embeddings_in
seq2seq.model.encoder.embedding.weight.requires_grad = False
seq2seq.model.decoder.embedding.weight.data = embeddings_out
seq2seq.model.decoder.embedding.weight.requires_grad = False

trainer = dl.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(seq2seq, train_loader)
#%% Test model performance

def unprocess(sequences, vocab, specials):
    """Convert numeric sequences to sentences."""
    sentences = []
    for sequence in sequences:
        idxs = sequence[sequence > len(specials) - 1]
        words = [vocab.lookup_token(idx) for idx in idxs]
        sentences.append(" ".join(words))
    return sentences

def translate(in_sentence, model, in_lang, in_vocab, out_vocab, specials):
    """Translate a sentence."""
    in_sentence = unicodedata.normalize("NFC", in_sentence)
    in_tokens = pad(tokenize(in_sentence, in_lang))
    in_sequence = (torch.tensor(in_vocab(in_tokens), dtype=torch.int)
                   .unsqueeze(0).to(next(model.parameters()).device))
    pred_sequence = model.evaluate(in_sequence)
    pred_sentence = unprocess(pred_sequence, out_vocab, specials)
    print(f"Predicted Translation: {pred_sentence[0]}\n")

in_sentence = "I bought a book."
translate(in_sentence, seq2seq.model, in_lang, in_vocab, out_vocab, specials)

in_sentence = "This book is very interesting."
translate(in_sentence, seq2seq.model, in_lang, in_vocab, out_vocab, specials)

in_sentence = "The book that I bought is very interesting."
translate(in_sentence, seq2seq.model, in_lang, in_vocab, out_vocab, specials)

#%% BLEU Score

from torchmetrics.text import BLEUScore

bleu_score = BLEUScore()

device = next(seq2seq.model.parameters()).device
for batch_index, (in_sequences, out_sequences) in enumerate(test_loader):
    in_sentences = unprocess(in_sequences.to(device), in_vocab, specials)
    pred_sequences = seq2seq.model.evaluate(in_sequences.to(device))
    pred_sentences = unprocess(pred_sequences, out_vocab, specials)
    out_sentences = unprocess(out_sequences.to(device), out_vocab, specials)

    bleu_score.update(pred_sentences, [[s] for s in out_sentences])

    print(f"Input Sentence: {in_sentences[0]}\n"
          + f"Predicted Translation: {pred_sentences[0]}\n"
          + f"Actual Translation: {out_sentences[0]}\n")
final_bleu = bleu_score.compute()
print(f"Validation BLEU Score: {final_bleu:.3f}")














