import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

class MultiModalProjection(nn.Module):
    def __init__(self, d1: int, d2: int, d3: int, d: int):
        super().__init__()
        self.W_ast = nn.Linear(d1, d)
        self.W_byte = nn.Linear(d2, d)
        self.W_cfg = nn.Linear(d3, d)
        self.layer_norm = nn.LayerNorm(d)
        
    def forward(self, ast_emb: torch.Tensor, byte_emb: torch.Tensor, cfg_emb: torch.Tensor) -> 'tuple[torch.Tensor]':
        H_ast = self.layer_norm(self.W_ast(ast_emb))
        H_byte = self.layer_norm(self.W_byte(byte_emb))
        H_cfg = self.layer_norm(self.W_cfg(cfg_emb))
        return H_ast, H_byte, H_cfg

class CoAttention(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        
        self.d = d
        self.scale = self.d ** -0.5
        
        # Query, Key, Value projections for all heads
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Linear projections and reshape for multi-head attention
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Combine values using attention weights
        context = torch.matmul(attention_probs, V)
        context = context.contiguous()
        
        return self.out_proj(context)

class MultiModalFusion(nn.Module):
    def __init__(self, d1: int, d2: int, d3: int, d: int = 64):
        super().__init__()
        self.projection = MultiModalProjection(d1, d2, d3, d)
        self.co_attention = CoAttention(d)
        
        # Final feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, d)
        )
        
        self.final_norm = nn.LayerNorm(d)
        
    def forward(self, ast_emb: torch.Tensor, byte_emb: torch.Tensor, cfg_emb: torch.Tensor, apply_attention: bool = True) -> torch.Tensor:
        # Project to common dimension
        H_ast, H_byte, H_cfg = self.projection(ast_emb, byte_emb, cfg_emb)
        
        if not apply_attention:
            return self.final_norm(H_ast + H_byte + H_cfg)
        
        # Compute cross-modal attention
        ast_byte = self.co_attention(H_ast, H_byte, H_byte)
        byte_ast = self.co_attention(H_byte, H_ast, H_ast)
        
        byte_cfg = self.co_attention(H_byte, H_cfg, H_cfg)
        cfg_byte = self.co_attention(H_cfg, H_byte, H_byte)
        
        ast_cfg = self.co_attention(H_ast, H_cfg, H_cfg)
        cfg_ast = self.co_attention(H_cfg, H_ast, H_ast)
        
        # Combine attended features
        E_delta = torch.cat([
            ast_byte + byte_ast,
            byte_cfg + cfg_byte,
            ast_cfg + cfg_ast
        ], dim=-1)
        
        # Final fusion through FFN
        E_delta = self.ffn(E_delta)
        
        # Residual connection and normalization
        E_final = self.final_norm(E_delta + H_ast + H_byte + H_cfg)
        
        return E_final

def clean_filenames(filenames, suffix: str) -> np.ndarray:
    """Clean filenames by removing suffixes and extensions"""
    if isinstance(filenames, pd.Series):
        # For CSV filenames, remove .sol extension
        filenames = [name.replace('.sol', '') for name in filenames]
    else:
        # For processed files, remove the specified suffix
        filenames = [name.replace(suffix, '') for name in filenames]
    return np.array(filenames)

def find_matching_files(csv_base_names: list, processed_files: list) -> tuple[list, list, dict]:
    """Find which CSV files have corresponding processed files"""
    matching_files = []
    matching_labels = []
    label_to_file = {}  # Map from base name to actual processed filename
    
    for csv_base_name in csv_base_names:
        # Find all processed files that start with this base name
        matches = [f for f in processed_files if f.startswith(csv_base_name)]
        if matches:
            # Take the first matching file
            matching_files.append(matches[0])
            matching_labels.append(csv_base_name)
            label_to_file[csv_base_name] = matches[0]
            print(f"Found match: {csv_base_name} -> {matches[0]}")
    
    return matching_files, matching_labels, label_to_file

def load_embeddings(ast_path: str, byte_path: str, cfg_path: str, labels_path: str) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]':
    # Load data
    ast_data = np.load(ast_path)
    byte_data = np.load(byte_path)
    cfg_data = np.load(cfg_path)
    labels = pd.read_csv(labels_path, header=0, dtype={'File': str, 'Label': int})
    
    # Get base names from CSV (without .sol extension)
    csv_base_names = clean_filenames(labels['File'], '')
    print(f"Number of files in CSV: {len(csv_base_names)}")
    print(f"First few CSV filenames: {csv_base_names[:5]}")
    
    # Get processed filenames
    ast_filenames = clean_filenames(ast_data['filenames'], '_ast_processed')
    byte_filenames = clean_filenames(byte_data['filenames'], '_assembly.txt')
    cfg_filenames = clean_filenames(cfg_data['filenames'], '_cfg_processed')
    
    print(f"Number of AST files: {len(ast_filenames)}")
    print(f"Number of bytecode files: {len(byte_filenames)}")
    print(f"Number of CFG files: {len(cfg_filenames)}")
    
    # Find matching files for each modality
    ast_matches, ast_labels, ast_label_to_file = find_matching_files(csv_base_names, ast_filenames)
    byte_matches, byte_labels, byte_label_to_file = find_matching_files(csv_base_names, byte_filenames)
    cfg_matches, cfg_labels, cfg_label_to_file = find_matching_files(csv_base_names, cfg_filenames)
    
    # Find intersection of all matches
    final_labels = sorted(set(ast_labels) & set(byte_labels) & set(cfg_labels))
    print(f"\nNumber of matching files found: {len(final_labels)}")
    
    # Create mapping from processed filename to index for each modality
    ast_idx = {name: i for i, name in enumerate(ast_filenames)}
    byte_idx = {name: i for i, name in enumerate(byte_filenames)}
    cfg_idx = {name: i for i, name in enumerate(cfg_filenames)}
    labels_idx = {name: i for i, name in enumerate(csv_base_names)}
    
    # Extract aligned data using the actual processed filenames
    ast_emb = np.array([ast_data['embeddings'][ast_idx[ast_label_to_file[name]]] for name in final_labels])
    byte_emb = np.array([byte_data['embeddings'][byte_idx[byte_label_to_file[name]]] for name in final_labels])
    cfg_emb = np.array([cfg_data['embeddings'][cfg_idx[cfg_label_to_file[name]]] for name in final_labels])
    labels_data = np.array([labels.loc[labels_idx[name], 'Label'] for name in final_labels])
    
    return ast_emb, byte_emb, cfg_emb, labels_data, final_labels

def main():
    # Paths
    ast_path = 'embeddings_ast/ast_embeddings.npz'
    byte_path = 'embeddings_bytecode/bytecode_embeddings.npz'
    cfg_path = 'embeddings_cfg/cfg_embeddings.npz'
    labels_path = 'dataset/labels.csv'
    
    output_dir = Path('attended_embeddings')
    output_dir.mkdir(exist_ok=True)

    embed_dim = 64
    
    # Load embeddings
    print("Loading embeddings...")
    ast_emb, byte_emb, cfg_emb, labels, filenames = load_embeddings(ast_path, byte_path, cfg_path, labels_path)
    
    # Check shapes before concatenation
    print(f"AST Embedding Shape: {ast_emb.shape}")
    print(f"Bytecode Embedding Shape: {byte_emb.shape}")
    print(f"CFG Embedding Shape: {cfg_emb.shape}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalFusion(
        d1=ast_emb.shape[1],
        d2=byte_emb.shape[1],
        d3=cfg_emb.shape[1],
        d=embed_dim
    ).to(device)
    
    # Convert to tensors
    ast_tensor = torch.FloatTensor(ast_emb).to(device)
    byte_tensor = torch.FloatTensor(byte_emb).to(device)
    cfg_tensor = torch.FloatTensor(cfg_emb).to(device)
    
    # Generate attended embeddings
    print("Generating attended embeddings...")
    model.eval()
    with torch.no_grad():
        attended_emb = model.forward(ast_tensor, byte_tensor, cfg_tensor)
        attended_emb = attended_emb.cpu().numpy()
    
    # Save attended embeddings
    output_path = output_dir / 'attended_embeddings.npz'
    np.savez(output_path,
             embeddings=attended_emb,
             labels=labels,
             filenames=filenames)
    
    print(f"Attended embeddings saved to {output_path}")
    print(f"Attended embedding shape: {attended_emb.shape}")

if __name__ == '__main__':
    main() 