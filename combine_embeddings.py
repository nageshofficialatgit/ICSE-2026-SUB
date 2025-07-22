import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

class MultiModalProjection(nn.Module):
    def __init__(self, d1, d2, d3, d4, d5, d6, d):
        super().__init__()
        self.W_ast = nn.Linear(d1, d)
        self.W_byte = nn.Linear(d2, d)
        self.W_cfg = nn.Linear(d3, d)
        self.W_ccfg = nn.Linear(d4, d)
        self.W_dfg = nn.Linear(d5, d)
        self.W_src = nn.Linear(d6, d)
        self.layer_norm = nn.LayerNorm(d)
    def forward(self, ast_emb, byte_emb, cfg_emb, ccfg_emb, dfg_emb, src_emb):
        H_ast = self.layer_norm(self.W_ast(ast_emb))
        H_byte = self.layer_norm(self.W_byte(byte_emb))
        H_cfg = self.layer_norm(self.W_cfg(cfg_emb))
        H_ccfg = self.layer_norm(self.W_ccfg(ccfg_emb))
        H_dfg = self.layer_norm(self.W_dfg(dfg_emb))
        H_src = self.layer_norm(self.W_src(src_emb))
        return H_ast, H_byte, H_cfg, H_ccfg, H_dfg, H_src

class CoAttention(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.scale = self.d ** -0.5
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)
        
    def forward(self, query, key, value):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, V)
        context = context.contiguous()
        return self.out_proj(context)

class MultiModalFusion(nn.Module):
    def __init__(self, d1, d2, d3, d4, d5, d6, d=64):
        super().__init__()
        self.projection = MultiModalProjection(d1, d2, d3, d4, d5, d6, d)
        self.co_attention = CoAttention(d)
        # 15 unique pairs for 6 modalities
        self.ffn = nn.Sequential(
            nn.Linear(d * 15, d * 6),
            nn.LayerNorm(d * 6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d * 6, d)
        )
        self.final_norm = nn.LayerNorm(d)
    def forward(self, ast_emb, byte_emb, cfg_emb, ccfg_emb, dfg_emb, src_emb, apply_attention=True):
        H_ast, H_byte, H_cfg, H_ccfg, H_dfg, H_src = self.projection(ast_emb, byte_emb, cfg_emb, ccfg_emb, dfg_emb, src_emb)
        if not apply_attention:
            return self.final_norm(H_ast + H_byte + H_cfg + H_ccfg + H_dfg + H_src)
        # All unique pairs (symmetric sum)
        modalities = [H_ast, H_byte, H_cfg, H_ccfg, H_dfg, H_src]
        attended = []
        for i in range(6):
            for j in range(i+1, 6):
                A, B = modalities[i], modalities[j]
                attended.append(self.co_attention(A, B, B) + self.co_attention(B, A, A))
        E_delta = torch.cat(attended, dim=-1)
        E_delta = self.ffn(E_delta)
        E_final = self.final_norm(E_delta + sum(modalities))
        return E_final

def clean_filenames(filenames, suffix: str) -> np.ndarray:
    if isinstance(filenames, pd.Series):
        filenames = [name.replace('.sol', '') for name in filenames]
    else:
        filenames = [name.replace(suffix, '') for name in filenames]
    return np.array(filenames)

def find_matching_files(csv_base_names: list, processed_files: list) -> tuple[list, list, dict]:
    matching_files = []
    matching_labels = []
    label_to_file = {}
    for csv_base_name in csv_base_names:
        matches = [f for f in processed_files if f.startswith(csv_base_name)]
        if matches:
            matching_files.append(matches[0])
            matching_labels.append(csv_base_name)
            label_to_file[csv_base_name] = matches[0]
    return matching_files, matching_labels, label_to_file

def load_embeddings(ast_path, byte_path, cfg_path, ccfg_path, dfg_path, src_path, labels_path):
    ast_data = np.load(ast_path)
    byte_data = np.load(byte_path)
    cfg_data = np.load(cfg_path)
    ccfg_data = np.load(ccfg_path)
    dfg_data = np.load(dfg_path)
    src_data = np.load(src_path)
    labels = pd.read_csv(labels_path, header=0, dtype={'File': str, 'Label': int})
    csv_base_names = clean_filenames(labels['File'], '')
    ast_filenames = clean_filenames(ast_data['filenames'], '_ast_processed')
    byte_filenames = clean_filenames(byte_data['filenames'], '_assembly.txt')
    cfg_filenames = clean_filenames(cfg_data['filenames'], '_cfg_processed')
    ccfg_filenames = clean_filenames(ccfg_data['filenames'], '_ccfg_processed')
    dfg_filenames = clean_filenames(dfg_data['filenames'], '_dfg_processed')
    src_filenames = clean_filenames(src_data['filenames'], '_sourcecode_processed')
    ast_matches, ast_labels, ast_label_to_file = find_matching_files(csv_base_names, ast_filenames)
    byte_matches, byte_labels, byte_label_to_file = find_matching_files(csv_base_names, byte_filenames)
    cfg_matches, cfg_labels, cfg_label_to_file = find_matching_files(csv_base_names, cfg_filenames)
    ccfg_matches, ccfg_labels, ccfg_label_to_file = find_matching_files(csv_base_names, ccfg_filenames)
    dfg_matches, dfg_labels, dfg_label_to_file = find_matching_files(csv_base_names, dfg_filenames)
    src_matches, src_labels, src_label_to_file = find_matching_files(csv_base_names, src_filenames)
    final_labels = sorted(set(ast_labels) & set(byte_labels) & set(cfg_labels) & set(ccfg_labels) & set(dfg_labels) & set(src_labels))
    ast_idx = {name: i for i, name in enumerate(ast_filenames)}
    byte_idx = {name: i for i, name in enumerate(byte_filenames)}
    cfg_idx = {name: i for i, name in enumerate(cfg_filenames)}
    ccfg_idx = {name: i for i, name in enumerate(ccfg_filenames)}
    dfg_idx = {name: i for i, name in enumerate(dfg_filenames)}
    src_idx = {name: i for i, name in enumerate(src_filenames)}
    labels_idx = {name: i for i, name in enumerate(csv_base_names)}
    ast_emb = np.array([ast_data['embeddings'][ast_idx[ast_label_to_file[name]]] for name in final_labels])
    byte_emb = np.array([byte_data['embeddings'][byte_idx[byte_label_to_file[name]]] for name in final_labels])
    cfg_emb = np.array([cfg_data['embeddings'][cfg_idx[cfg_label_to_file[name]]] for name in final_labels])
    ccfg_emb = np.array([ccfg_data['embeddings'][ccfg_idx[ccfg_label_to_file[name]]] for name in final_labels])
    dfg_emb = np.array([dfg_data['embeddings'][dfg_idx[dfg_label_to_file[name]]] for name in final_labels])
    src_emb = np.array([src_data['embeddings'][src_idx[src_label_to_file[name]]] for name in final_labels])
    labels_data = np.array([labels.loc[labels_idx[name], 'Label'] for name in final_labels])
    return ast_emb, byte_emb, cfg_emb, ccfg_emb, dfg_emb, src_emb, labels_data, final_labels 