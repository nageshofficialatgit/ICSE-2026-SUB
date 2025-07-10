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

torch.set_default_device(torch.device('cpu'))
device = torch.device('cuda')

# Custom Tokenizer for Solidity Bytecode
class SolidityTokenizer:
    def __init__(self):
        self.opcodes = {
            'add': 1, 'mul': 2, 'sub': 3, 'div': 4, 'sdiv': 5, 'mod': 6, 'smod': 7,
            'addmod': 8, 'mulmod': 9, 'exp': 10, 'signextend': 11, 'lt': 12, 'gt': 13,
            'slt': 14, 'sgt': 15, 'eq': 16, 'iszero': 17, 'and': 18, 'or': 19, 'xor': 20,
            'not': 21, 'byte': 22, 'shl': 23, 'shr': 24, 'sar': 25, 'keccak256': 26, 'address': 27,
            'balance': 28, 'origin': 29, 'caller': 30, 'callvalue': 31, 'calldataload': 32,
            'calldatasize': 33, 'calldatacopy': 34, 'codesize': 35, 'codecopy': 36, 'gasprice': 37,
            'extcodesize': 38, 'extcodecopy': 39, 'returndatasize': 40, 'returndatacopy': 41,
            'blockhash': 42, 'coinbase': 43, 'timestamp': 44, 'number': 45, 'difficulty': 46,
            'gaslimit': 47, 'chainid': 48, 'selfbalance': 49, 'basefee': 50, 'pop': 51,
            'mload': 52, 'mstore': 53, 'mstore8': 54, 'sload': 55, 'sstore': 56, 'jump': 57,
            'jumpi': 58, 'pc': 59, 'msize': 60, 'gas': 61, 'jumpdest': 62, 'push1': 63,
            'push2': 64, 'push3': 65, 'push4': 66, 'push5': 67, 'push6': 68, 'push7': 69,
            'push8': 70, 'push9': 71, 'push10': 72, 'push11': 73, 'push12': 74, 'push13': 75,
            'push14': 76, 'push15': 77, 'push16': 78, 'push17': 79, 'push18': 80, 'push19': 81,
            'push20': 82, 'push21': 83, 'push22': 84, 'push23': 85, 'push24': 86, 'push25': 87,
            'push26': 88, 'push27': 89, 'push28': 90, 'push29': 91, 'push30': 92, 'push31': 93,
            'push32': 94, 'dup1': 95, 'dup2': 96, 'dup3': 97, 'dup4': 98, 'dup5': 99, 'dup6': 100,
            'dup7': 101, 'dup8': 102, 'dup9': 103, 'dup10': 104, 'dup11': 105, 'dup12': 106,
            'dup13': 107, 'dup14': 108, 'dup15': 109, 'dup16': 110, 'swap1': 111, 'swap2': 112,
            'swap3': 113, 'swap4': 114, 'swap5': 115, 'swap6': 116, 'swap7': 117, 'swap8': 118,
            'swap9': 119, 'swap10': 120, 'swap11': 121, 'swap12': 122, 'swap13': 123,
            'swap14': 124, 'swap15': 125, 'swap16': 126, 'log0': 127, 'log1': 128, 'log2': 129,
            'log3': 130, 'log4': 131, 'create': 132, 'call': 133, 'callcode': 134, 'return': 135,
            'delegatecall': 136, 'create2': 137, 'staticcall': 138, 'revert': 139, 'invalid': 140,
            'selfdestruct': 141, 'stop': 142, 'tag': 143, '{': 144, '}': 145, '(': 146, ')': 147,
            'dataOffset': 148, 'dataSize': 149, 'sub_0': 150, 'sub_1':151, 'assembly': 152, 'calldatasize': 153,
            ',': 154, 'deployTimeAddress': 155, 'bytecodeSize': 156, 'sub_2': 157, 'sub_3': 158, 'linkerSymbol': 159
        }
        self.identifiers = {}
        self.unknown = {}
        self.pad_token_id = 0 # For padding
        self.mask_token_id = len(self.opcodes) + 1  # For masking
        self.identifier_start_id = len(self.opcodes) + 2  # Start ID for identifiers

    def tokenize(self, bytecode):
        tokens = []
        for op in bytecode.split():
            if op in self.opcodes:
                tokens.append(self.opcodes[op])
            elif op.startswith("0x") or op.startswith("_") or op.isnumeric():  # Identifier detection
                if op in self.identifiers:
                    tokens.append(self.identifiers[op])
                else:
                    tokens.append(self.identifier_start_id)  # Assign ID for identifiers
                    self.identifiers[op] = self.identifier_start_id
                    self.identifier_start_id += 1  # Increment for unique identifier IDs
            elif op in self.unknown:
                tokens.append(self.unknown[op])
            else:
                tokens.append(self.identifier_start_id)  # Assign ID for identifiers
                self.unknown[op] = self.identifier_start_id
                self.identifier_start_id += 1  # Increment for unique identifier IDs
        return tokens

    def pad(self, tokens, max_length):
        return tokens[:max_length] + [self.pad_token_id] * max(0, max_length - len(tokens))

class SolidityBytecodeDataset(Dataset):
    def __init__(self, bin_folder_path, tokenizer, max_seq_len):
        self.bytecodes, self.filenames = self.load_bytecode_files(bin_folder_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @staticmethod
    def load_bytecode_files(folder_path):
        bytecodes = []
        filenames = []
        for root, _, files in os.walk(os.path.dirname(folder_path)):
            for file in files:
                if file.endswith(".txt"):
                    filenames.append(file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        bytecodes.append(f.read().strip())
        return bytecodes, filenames

    def __len__(self):
        return len(self.bytecodes)

    def __getitem__(self, idx):
        bytecode = self.bytecodes[idx]
        tokens = self.tokenizer.tokenize(bytecode)
        tokens = self.tokenizer.pad(tokens, self.max_seq_len)
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
        return torch.tensor(tokens), torch.tensor(attention_mask)

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
    output_path = 'embeddings_bytecode/bytecode_embeddings.npz'
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
    plt.savefig('embedding_bytecode/training_loss.png')
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
    
    plt.title('t-SNE visualization of  embeddings')
    plt.savefig('embeddings_bytecode/embeddings_tsne.png')
    plt.close()

if __name__ == "__main__":
    bin_folder_path = "Bytecode/Dataset_Assembly/"
    vocab_size = 18432  # Number of opcodes + special tokens
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

    tokenizer = SolidityTokenizer()

    dataset = SolidityBytecodeDataset(bin_folder_path, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLMTransformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, d_ff, dropout).to(device)
    model.train()

    optimizer = torch.optim.AdamW([{'params': model.module_list.parameters()},
                                   {'params': model.projection.parameters(), 'lr': 1e-4}], lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)


    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with device:
                # Clone input_ids for masking and labels
                masked_input_ids, labels = input_ids.clone(), input_ids.clone()
                
                # Generate random mask
                rand = torch.rand(input_ids.shape)
                mask = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
                masked_input_ids[mask] = tokenizer.mask_token_id  # Replace with [MASK] token
                labels[~mask] = -100  # Set non-masked positions to ignore_index (-100)

                # Forward pass
                logits, embeddings = model(masked_input_ids, attention_mask)

                # Reshape logits and labels for loss computation
                logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
                labels = labels.view(-1)  # [batch_size * seq_len]

                # Compute loss
                loss = loss_fn(logits, labels).to(device) + model.contrastive_loss(embeddings).to(device)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        # Log epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print('new identifies', tokenizer.identifiers)
    print('new unknown', tokenizer.unknown)
    print('total tokens', len(tokenizer.opcodes) + len(tokenizer.identifiers) + len(tokenizer.unknown))

# model.embedding.parameters()
# model.position_embedding.parameters()
    # Path to folder containing new bytecode files
    bytecode_folder_path = "Bytecode/Dataset_Assembly/"  # Replace with the actual folder path

    # Generate embeddings for new bytecodes
    embeddings = generate_embeddings(bytecode_folder_path, model, tokenizer, max_seq_len)

    # Store embeddings


