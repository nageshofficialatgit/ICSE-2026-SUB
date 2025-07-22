import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
import re

torch.set_default_device(torch.device('cpu'))
device = torch.device('cuda')

# New Tokenizer for Solidity Source Code using Word2Vec
class SourceCodeWord2VecTokenizer:
    def __init__(self, sol_folder_path, embed_dim=256, min_count=1, window=5):
        self.sol_folder_path = sol_folder_path
        self.embed_dim = embed_dim
        self.min_count = min_count
        self.window = window
        self.pad_token = '<PAD>'
        self.mask_token = '<MASK>'
        self.special_tokens = [self.pad_token, self.mask_token]
        self.model = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.pad_token_id = 0
        self.mask_token_id = 1
        self._train_word2vec()

    def simple_tokenize(self, code):
        # Splits on words and non-whitespace symbols
        return re.findall(r"\w+|[^\w\s]", code)

    def _get_all_tokens(self):
        tokens = []
        for root, _, files in os.walk(self.sol_folder_path):
            for file in files:
                if file.endswith('.sol'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                        tokens.append(self.simple_tokenize(code))
        return tokens

    def _train_word2vec(self):
        sentences = self._get_all_tokens()
        self.model = Word2Vec(sentences, vector_size=self.embed_dim, window=self.window, min_count=self.min_count, workers=4)
        self.vocab = list(self.model.wv.index_to_key)
        self.word2idx = {word: idx+len(self.special_tokens) for idx, word in enumerate(self.vocab)}
        self.word2idx[self.pad_token] = self.pad_token_id
        self.word2idx[self.mask_token] = self.mask_token_id
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def tokenize(self, code):
        return self.simple_tokenize(code)

    def convert_tokens_to_ids(self, tokens):
        return [self.word2idx.get(token, self.mask_token_id) for token in tokens]

    def pad(self, tokens, max_length):
        return tokens[:max_length] + [self.pad_token_id] * max(0, max_length - len(tokens))

class SoliditySourceCodeDataset(Dataset):
    def __init__(self, sol_folder_path, tokenizer, max_seq_len):
        self.codes, self.filenames = self.load_solidity_files(sol_folder_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @staticmethod
    def load_solidity_files(folder_path):
        codes = []
        filenames = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.sol'):
                    filenames.append(file)
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        codes.append(f.read())
        return codes, filenames

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        tokens = self.tokenizer.tokenize(code)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.pad(token_ids, self.max_seq_len)
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in token_ids]
        return torch.tensor(token_ids), torch.tensor(attention_mask)

class MLMTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, d_ff, dropout=0.1):
        super(MLMTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

        self.module_list = nn.ModuleList([
            self.embedding,
            self.position_embedding,
            self.encoder,
            self.fc
        ])

        self.projection = nn.Linear(max_seq_len, 1)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with device:
            seq_len = input_ids.size(1)
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

            x = self.embedding(input_ids) + self.position_embedding(positions)
            y = self.projection(x.permute(0, 2, 1))

            src_key_padding_mask = ~attention_mask.bool()

            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

            x = self.fc(x)
            return F.softmax(x, dim=-1), F.tanh(y.view(-1, embed_dim))
    
    def contrastive_loss(self, embeddings: torch.Tensor, temperature: float = 0.5):
        """Compute contrastive loss for the embeddings"""
        embeddings = embeddings.to(device)
        with device:
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(embeddings, embeddings.t())
            
            # Mask out self-similarity
            mask = torch.eye(embeddings.shape[0])
            mask = 1 - mask
            
            # Compute loss
            similarity_matrix = similarity_matrix * mask
            positives = similarity_matrix.diag()
            negatives = similarity_matrix.sum(dim=1) - positives
            
            loss = -torch.log(torch.exp(positives / temperature) / 
                            torch.exp(negatives / temperature).sum(dim=0))
            return loss.mean()

def generate_embeddings(bytecode_folder_path, model, tokenizer, max_seq_len):
    """
    Generate embeddings for bytecodes in the specified folder using the model's embedding layers.

    Args:
        bytecode_folder_path (str): Path to the folder containing bytecode files.
        model (MLMTransformer): The trained model with embedding layers.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing bytecodes.
        max_seq_len (int): Maximum sequence length for tokenization.

    Returns:
        List of numpy arrays containing embeddings for each bytecode.
    """
    # Load bytecodes from the specified folder
    bytecodes, filenames = SolidityBytecodeDataset.load_bytecode_files(bytecode_folder_path)
    
    # Prepare a list to store embeddings
    embeddings_list = []

    for bytecode in tqdm(bytecodes, desc="Generating embeddings"):
        # Tokenize the bytecode
        tokens = tokenizer.tokenize(bytecode)
        input_ids = tokenizer.pad(tokens, max_seq_len)
        input_ids = torch.tensor(input_ids).to(device)

        # Generate embeddings using the model's embedding layers
        with torch.no_grad():
            with device:
                seq_len = input_ids.size(0)
                positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

                # Get embeddings
                token_embeddings = model.embedding(input_ids)
                position_embeddings = model.position_embedding(positions)

                # Sum token and position embeddings
                embeddings = token_embeddings + position_embeddings

                embeddings = F.tanh(model.projection(embeddings.permute(0, 2, 1)).view(-1, embed_dim))
                
                # Append to the list
                embeddings_list.append(embeddings.squeeze(0).cpu().numpy())
    embeddings_list = np.vstack(embeddings_list)

    save_embeddings(embeddings_list, filenames)
    visualize_embeddings(embeddings_list, filenames)

    return embeddings_list

def save_embeddings(embeddings: np.ndarray, filenames: list[str]):
    """Save embeddings to file"""
    output_path = 'embeddings_sourcecode/sourcecode_embeddings.npz'
    np.savez(output_path, 
                embeddings=embeddings,
                filenames=filenames)
    print(f"Embeddings saved to {output_path}")

def plot_training_loss(losses: list[float]):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('embeddings_sourcecode/training_loss.png')
    plt.close()

def visualize_embeddings(embeddings: np.ndarray, filenames: list[str]):
    """Visualize embeddings using t-SNE"""
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Add labels for some points
    for i, filename in enumerate(filenames):
        if i % max(1, len(filenames) // 20) == 0:  # Label every nth point
            plt.annotate(filename, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('t-SNE visualization of source code embeddings')
    plt.savefig('embeddings_sourcecode/embeddings_tsne.png')
    plt.close()

if __name__ == "__main__":
    sol_folder_path = "dataset/codes/"
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    max_seq_len = 2048
    d_ff = 128
    batch_size = 4
    dropout = 0.1
    learning_rate = 1e-3
    num_epochs = 20
    mask_prob = 0.15

    tokenizer = SourceCodeWord2VecTokenizer(sol_folder_path, embed_dim=embed_dim)
    vocab_size = len(tokenizer.word2idx)

    dataset = SoliditySourceCodeDataset(sol_folder_path, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLMTransformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, d_ff, dropout).to(device)
    model.train()

    optimizer = torch.optim.AdamW([{'params': model.module_list.parameters()},
                                   {'params': model.projection.parameters(), 'lr': 1e-4}], lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with device:
                masked_input_ids, labels = input_ids.clone(), input_ids.clone()
                rand = torch.rand(input_ids.shape)
                mask = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
                masked_input_ids[mask] = tokenizer.mask_token_id
                labels[~mask] = -100
                logits, embeddings = model(masked_input_ids, attention_mask)
                logits = logits.view(-1, vocab_size)
                labels = labels.view(-1)
                loss = loss_fn(logits, labels).to(device) + model.contrastive_loss(embeddings).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print('Tokenizer vocab size:', vocab_size)
    print('Tokenizer word2idx:', list(tokenizer.word2idx.items())[:10])

    def save_embeddings(embeddings: np.ndarray, filenames: list[str]):
        output_path = 'embeddings_sourcecode/sourcecode_embeddings.npz'
        np.savez(output_path, embeddings=embeddings, filenames=filenames)
        print(f"Embeddings saved to {output_path}")

    def plot_training_loss(losses: list[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('embeddings_sourcecode/training_loss.png')
        plt.close()

    def visualize_embeddings(embeddings: np.ndarray, filenames: list[str]):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        for i, filename in enumerate(filenames):
            if i % max(1, len(filenames) // 20) == 0:
                plt.annotate(filename, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        plt.title('t-SNE visualization of source code embeddings')
        plt.savefig('embeddings_sourcecode/embeddings_tsne.png')
        plt.close()

    def generate_embeddings(sol_folder_path, model, tokenizer, max_seq_len):
        codes, filenames = SoliditySourceCodeDataset.load_solidity_files(sol_folder_path)
        embeddings_list = []
        for code in tqdm(codes, desc="Generating embeddings"):
            tokens = tokenizer.tokenize(code)
            input_ids = tokenizer.pad(tokenizer.convert_tokens_to_ids(tokens), max_seq_len)
            input_ids = torch.tensor(input_ids).to(device)
            with torch.no_grad():
                with device:
                    seq_len = input_ids.size(0)
                    positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
                    token_embeddings = model.embedding(input_ids)
                    position_embeddings = model.position_embedding(positions)
                    embeddings = token_embeddings + position_embeddings
                    embeddings = F.tanh(model.projection(embeddings.permute(0, 2, 1)).view(-1, embed_dim))
                    embeddings_list.append(embeddings.squeeze(0).cpu().numpy())
        embeddings_list = np.vstack(embeddings_list)
        save_embeddings(embeddings_list, filenames)
        visualize_embeddings(embeddings_list, filenames)
        return embeddings_list

    plot_training_loss(losses)
    embeddings = generate_embeddings(sol_folder_path, model, tokenizer, max_seq_len)


