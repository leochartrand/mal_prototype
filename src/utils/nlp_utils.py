import torch
import numpy as np

def tokenize_commands(commands_txt):
    return [cmd.split() for cmd in commands_txt]

def build_vocab(tokenized_commands):
    vocab = set()
    vocab.add('<PAD>')  # Padding token
    for cmd in tokenized_commands:
        for word in cmd:
            vocab.add(word)
    return sorted(vocab)

def create_mappings(vocab):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def index_encode_commands(tokenized_commands, word2idx):
    encoded_commands = []
    for cmd in tokenized_commands:
        encoded = [word2idx[word] for word in cmd]
        encoded_commands.append(encoded)
    return encoded_commands

def one_hot_encode_commands(encoded_commands, vocab):
    commands = []
    for encoded in encoded_commands:
        one_hot = torch.zeros(len(vocab))
        for idx in encoded:
            one_hot[idx] = 1
        commands.append(one_hot)
    commands = torch.stack(commands)  # [N, vocab_size]
    return commands

def pad_commands(encoded_commands, pad_idx, max_length=4):    
    # Create padded tensor
    padded_tensor = torch.full(
        (len(encoded_commands), max_length), 
        pad_idx, 
        dtype=torch.long
    )
    # Fill in actual tokens
    for i, cmd in enumerate(encoded_commands):
        seq_len = min(len(cmd), max_length)  # Truncate if longer than max_length
        padded_tensor[i, :seq_len] = torch.tensor(cmd[:seq_len], dtype=torch.long)
    
    return padded_tensor

def decode_commands(encoded_commands, idx2word):
    decoded_commands = []
    for encoded in encoded_commands:
        # one-hot to indices, get all indices with value 1
        if isinstance(encoded, torch.Tensor):
            encoded = torch.nonzero(encoded).squeeze().tolist()
        if isinstance(encoded, int):
            encoded = [encoded]
        if not isinstance(encoded, list):
            encoded = list(encoded)
        encoded = [idx2word[idx] for idx in encoded if idx in idx2word]
        decoded = ' '.join(encoded)
        decoded_commands.append(decoded)
    return decoded_commands

def prepare_fewhot_commands(commands_txt):
    tokenized_commands = tokenize_commands(commands_txt)
    vocab = build_vocab(tokenized_commands)
    word2idx, idx2word = create_mappings(vocab)
    encoded_commands = index_encode_commands(tokenized_commands, word2idx)
    commands = one_hot_encode_commands(encoded_commands, vocab)
    return commands, vocab, word2idx, idx2word

def prepare_padded_commands(commands_txt):
    tokenized_commands = tokenize_commands(commands_txt)
    vocab = build_vocab(tokenized_commands)
    word2idx, idx2word = create_mappings(vocab)
    encoded_commands = index_encode_commands(tokenized_commands, word2idx)
    commands = pad_commands(encoded_commands, pad_idx=0, max_length=4) 
    return commands, vocab, word2idx, idx2word

def encode_commands(commands_txt, word2idx, vocab):
    tokenized_commands = tokenize_commands(commands_txt)
    encoded_commands = index_encode_commands(tokenized_commands, word2idx)
    commands = one_hot_encode_commands(encoded_commands, vocab)
    return commands