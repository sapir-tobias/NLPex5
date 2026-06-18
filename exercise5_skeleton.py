"""
Exercise 5 – Sentiment Analysis  |  SKELETON
=============================================
Fill in all functions/methods marked with  # TODO
Do NOT change function signatures unless you document the change in your README.
"""

import os
import pickle
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import data_loader
from data_loader import get_negated_polarity_examples, get_rare_words_examples

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 52

ONEHOT_AVERAGE       = "onehot_average"
TRANSFORMER_AVERAGE  = "transformer_average"
TRANSFORMER_SEQUENCE = "transformer_sequence"

TRAIN = "train"
VAL   = "val"
TEST  = "test"

TRANSFORMER_MODEL_NAME = "distilroberta-base"


# ──────────────────────────────────────────────────────────────────────────────
# Device helper  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def get_available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Pickle helpers  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Transformer embeddings loader
# ──────────────────────────────────────────────────────────────────────────────

def load_transformer_embeddings(cache_path="transformer_emb_cache.pkl"):
    """
    Loads distilroberta-base and extracts the token embedding matrix.
    Caches the matrix locally to avoid reloading.

    Returns
    -------
    tokenizer        : HuggingFace tokenizer
    embedding_matrix : np.ndarray of shape (vocab_size, hidden_dim)
    embedding_dim    : int
    """
    # If already cached, load from disk
    if os.path.exists(cache_path):
        tokenizer, embedding_matrix, embedding_dim = load_pickle(cache_path)
        return tokenizer, embedding_matrix, embedding_dim

    # If not cached, load from HuggingFace and cache
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)

    # Extract the embedding matrix from the model
    embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    embedding_dim = embedding_matrix.shape[1]

    # Save to cache for future use
    save_pickle((tokenizer, embedding_matrix, embedding_dim), cache_path)

    return tokenizer, embedding_matrix, embedding_dim


# ──────────────────────────────────────────────────────────────────────────────
# One-hot helpers  (Section 6)
# ──────────────────────────────────────────────────────────────────────────────

def get_word_to_ind(words_list):
    """
    Returns a dict mapping each word in words_list to a unique integer index.
    :param words_list: list of words
    :return: dict {word: index}
    """
    dict = {word: i for i, word in enumerate(words_list)}
    return dict


def get_one_hot(size, ind):
    """
    Returns a one-hot numpy vector of length `size` with 1 at position `ind`.
    :param size: length of the vector
    :param ind: index of the 1 entry
    :return: np.ndarray of shape (size,)
    """
    vector = np.zeros(size)
    vector[ind] = 1
    return vector


def average_one_hots(sent, word_to_ind):
    """
    Returns the average one-hot embedding for the tokens in `sent`.
    Words not in word_to_ind are ignored (not added to the average).
    :param sent: Sentence object (sent.text is a list of strings)
    :param word_to_ind: dict {word: index}
    :return: np.ndarray of shape (vocab_size,)
    """
    words = sent.text
    one_hots = []
    for word in words:
        if word in word_to_ind:
            ind = word_to_ind[word]
            one_hots.append(get_one_hot(len(word_to_ind), ind))
    if not one_hots:
        return np.zeros(len(word_to_ind))
    return np.mean(one_hots, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer-embedding helpers  (Sections 7 & 8)
# ──────────────────────────────────────────────────────────────────────────────

def get_transformer_average(sent, tokenizer, embedding_matrix, embedding_dim):
    """
    Tokenizes `sent` and returns the average of its Transformer token embeddings.
    Special tokens (e.g. <s>, </s>) should be excluded.
    Token IDs outside the embedding matrix range are treated as unknown (skip them).

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param embedding_dim:    int
    :return: np.ndarray of shape (embedding_dim,)
    """
    # tokenize the sentence
    tokens = tokenizer.tokenize(" ".join(sent.text))
    # convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # filter out special tokens and unknown IDs
    valid_embeddings = []
    for token_id in token_ids:
        if 0 <= token_id < embedding_matrix.shape[0]:
            valid_embeddings.append(embedding_matrix[token_id])
    if not valid_embeddings:
        return np.zeros(embedding_dim)
    return np.mean(valid_embeddings, axis=0)        


def sentence_to_embedding(sent, tokenizer, embedding_matrix, seq_len, embedding_dim):
    """
    Maps `sent` to a fixed-size matrix of Transformer token embeddings.
    - Sentences longer than seq_len are truncated (keep the first seq_len tokens).
    - Sentences shorter than seq_len are padded with zero-vectors.
    Special tokens are excluded from the encoding.

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param seq_len:          int – target sequence length
    :param embedding_dim:    int
    :return: np.ndarray of shape (seq_len, embedding_dim)
    """
    # Extract the set of special token IDs from the tokenizer
    special_ids = set(tokenizer.all_special_ids)

    # Encode the sentence into token IDs
    token_ids = tokenizer.encode(sent.text if isinstance(sent.text, str) else " ".join(sent.text))
    # Filter out special tokens
    token_ids = [tid for tid in token_ids if tid not in special_ids]

    # Initialize the result matrix with zeros
    result = np.zeros((seq_len, embedding_dim), dtype=np.float32)

    # Drop everything beyond seq_len
    token_ids = token_ids[:seq_len]

    # Insert the corresponding embeddings for valid token IDs
    for i, tid in enumerate(token_ids):
        if tid < len(embedding_matrix):
            result[i] = embedding_matrix[tid]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataManager  (provided)
# ──────────────────────────────────────────────────────────────────────────────

class OnlineDataset(Dataset):
    """PyTorch Dataset that converts sentences on-the-fly."""

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent  = self.data[idx]
        emb   = self.sent_func(sent, **self.sent_func_kwargs)
        label = sent.sentiment_class
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class DataManager:
    """
    Handles all data management.
    data_type: one of ONEHOT_AVERAGE, TRANSFORMER_AVERAGE, TRANSFORMER_SEQUENCE
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=64,
                 embedding_dim=None, tokenizer=None, embedding_matrix=None):

        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()
        self.sentences[VAL]  = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        words_list = list(self.sentiment_dataset.get_word_counts().keys())

        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}

        elif data_type == TRANSFORMER_AVERAGE:
            assert tokenizer is not None and embedding_matrix is not None
            self.sent_func = get_transformer_average
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "embedding_dim":    embedding_dim,
            }

        elif data_type == TRANSFORMER_SEQUENCE:
            assert tokenizer is not None and embedding_matrix is not None
            self.sent_func = sentence_to_embedding
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "seq_len":          SEQ_LEN,
                "embedding_dim":    embedding_dim,
            }
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        self.torch_datasets  = {k: OnlineDataset(sents, self.sent_func, self.sent_func_kwargs)
                                 for k, sents in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == TRAIN))
                                 for k, ds in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        return np.array([s.sentiment_class for s in self.sentences[data_subset]])

    def get_input_shape(self):
        return self.torch_datasets[TRAIN][0][0].shape


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class LogLinear(nn.Module):
    """
    Log-linear model for sentiment analysis.
    Used for both the one-hot model (Section 6) and the Transformer-embedding model (Section 7).
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass

    def predict(self, x):
        # TODO
        pass


class LSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment analysis (Section 8).
    Input  : (batch, seq_len, embedding_dim)
    Output : logits of shape (batch,)
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        # TODO

    def forward(self, text):
        # TODO
        pass

    def predict(self, text):
        # TODO
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def binary_accuracy(preds, y):
    """
    Returns the fraction of correctly classified examples.
    preds: probabilities in [0,1] (or logits – round at 0.5)
    y:     true labels {0, 1}
    :return: float
    """
    # Round the preds
    rounded_preds = torch.round(preds)
    # Compare with true labels
    correct = (rounded_preds == y).float()
    # Return the accuracy
    return correct.sum() / len(correct)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    Runs one full pass over the training data.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader for the training set
    :param optimizer:     torch optimizer
    :param criterion:     loss criterion (BCEWithLogitsLoss)
    :return: (avg_loss, avg_accuracy) over the epoch
    """
    model.train() # Set the model to training mode
    total_loss = 0.0
    total_acc = 0.0
    for inputs, labels in data_iterator:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)  # Ensure outputs are of shape (batch,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += binary_accuracy(torch.sigmoid(outputs), labels)
    avg_loss = total_loss / len(data_iterator)
    avg_acc = total_acc / len(data_iterator)
    return avg_loss, avg_acc


def evaluate(model, data_iterator, criterion):
    """
    Evaluates the model WITHOUT updating parameters.
    Use torch.no_grad() to avoid computing unnecessary gradients.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader (val or test)
    :param criterion:     loss criterion
    :return: (avg_loss, avg_accuracy)
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for inputs, labels in data_iterator:
                outputs = model(inputs).squeeze(-1) # Ensure outputs are of shape (batch,)
                loss = criterion(outputs, labels.float())
        total_loss += loss.item()
        total_acc += binary_accuracy(torch.sigmoid(outputs), labels)
    avg_loss = total_loss / len(data_iterator)
    avg_acc = total_acc / len(data_iterator)
    return avg_loss, avg_acc


def get_predictions_for_data(model, data_iter):
    """
    Collects all model predictions (probabilities) over a data iterator.
    :param model:     a LogLinear or LSTM model
    :param data_iter: PyTorch DataLoader
    :return: np.ndarray of shape (n_examples,)
    """
    pass


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.0):
    """
    Full training loop using Adam optimizer and BCEWithLogitsLoss.
    Records per-epoch train/validation loss and accuracy.

    :param model:        LogLinear or LSTM instance
    :param data_manager: DataManager instance
    :param n_epochs:     number of epochs
    :param lr:           learning rate
    :param weight_decay: L2 regularization coefficient
    :return: dict with keys train_loss, train_acc, val_loss, val_acc
             (each a list of length n_epochs)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion)
        val_loss, val_acc = evaluate(model, data_manager.get_torch_iterator(VAL), criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())
        
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper  (provided – feel free to modify)
# ──────────────────────────────────────────────────────────────────────────────

def plot_and_save(history, title_prefix, save_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} – Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_acc.png")
    plt.close()


def evaluate_special_subsets(model, data_manager):
    """Prints accuracy on negated-polarity and rare-words subsets (provided)."""
    dataset    = data_manager.sentiment_dataset
    test_sents = data_manager.sentences[TEST]

    neg_idx  = get_negated_polarity_examples(test_sents)
    rare_idx = get_rare_words_examples(test_sents, dataset)

    all_preds  = get_predictions_for_data(model, data_manager.get_torch_iterator(TEST))
    all_labels = data_manager.get_labels(TEST)

    for name, indices in [("Negated polarity", neg_idx), ("Rare words", rare_idx)]:
        if len(indices) == 0:
            print(f"  {name}: no examples found")
            continue
        acc = float(np.mean(
            (all_preds[indices] >= 0.5).astype(float) == all_labels[indices]
        ))
        print(f"  {name} accuracy ({len(indices)} examples): {acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Log-linear with one-hot
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_one_hot():
    """
    Creates all objects needed for training/evaluation of the one-hot log-linear model
    and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    """
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Log-linear with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_transformer():
    """
    Creates all objects needed for training/evaluation of the Transformer-embedding
    log-linear model and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    """
    # TODO
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 – Bi-LSTM with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_with_transformer():
    """
    Creates all objects needed for training/evaluation of the bi-LSTM model
    and runs the training process.
    Hyperparameters: lr=0.001, weight_decay=0.0001, dropout=0.5, n_epochs=4, batch_size=64
    """
    # TODO
    pass


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_log_linear_with_one_hot()
    # train_log_linear_with_transformer()
    # train_lstm_with_transformer()


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 – Fine-Tuning a Transformer Model
# ══════════════════════════════════════════════════════════════════════════════

class TransformerSentimentDataset(Dataset):
    """
    Encodes each sentence with the Transformer tokenizer on the fly.
    __getitem__ should return (input_ids, attention_mask, label).
    """

    MAX_LENGTH = 64

    def __init__(self, sentences, tokenizer, max_length=None):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.max_length = max_length or self.MAX_LENGTH

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # TODO
        pass


class TransformerSentimentClassifier(nn.Module):
    """
    distilroberta-base base model (no HF classification head) +
    a single Linear layer on top of the [CLS] token's final hidden state.
    forward() returns logits (no sigmoid).
    predict() returns probabilities (after sigmoid).
    """

    def __init__(self, model_name=TRANSFORMER_MODEL_NAME):
        super().__init__()
        # TODO

    def forward(self, input_ids, attention_mask):
        # TODO
        pass

    def predict(self, input_ids, attention_mask):
        # TODO
        pass


def train_transformer(dataset_path="stanfordSentimentTreebank",
                      batch_size=64, n_epochs=2, lr=1e-5, weight_decay=0.0):
    """
    Full training procedure for the fine-tuned Transformer.
    Use BCEWithLogitsLoss and Adam. No HuggingFace Trainer.
    Report train/val loss and accuracy per epoch, test loss/accuracy,
    and accuracy on the two special subsets.
    """
    # TODO
    pass
