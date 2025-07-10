import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple

def clean_filenames(filenames, suffixes: list) -> np.ndarray:
    """Remove specified suffixes from filenames."""
    for suffix in suffixes:
        filenames = [name.replace(suffix, '') for name in filenames]
    return np.array(filenames)

def load_embeddings(cfg_path: str, ast_path: str, byte_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load embeddings from all three sources and reorder them based on label filenames."""
    cfg_data = np.load(cfg_path)
    ast_data = np.load(ast_path)
    byte_data = np.load(byte_path)
    
    # Create labels array (same for all samples)
    labels = pd.read_csv(labels_path, header=0, dtype=str)
    
    # Clean filenames
    cfg_filenames = clean_filenames(cfg_data['filenames'], ['_cfg_processed'])
    ast_filenames = clean_filenames(ast_data['filenames'], ['_ast_processed'])
    byte_filenames = clean_filenames(byte_data['filenames'], ['_assembly.txt'])
    labels_filenames = clean_filenames(labels.File, [''])
    
    # Sort filenames for comparison
    cfg_filenames.sort()
    ast_filenames.sort()
    byte_filenames.sort()
    labels_filenames.sort()
    
    # Print cleaned and sorted filenames for debugging
    print("Sorted CFG Filenames:", cfg_filenames)
    print("Sorted AST Filenames:", ast_filenames)
    print("Sorted Bytecode Filenames:", byte_filenames)
    print("Sorted Labels Filenames:", labels_filenames)

    final_filenames = np.intersect1d(cfg_filenames, np.intersect1d(ast_filenames, np.intersect1d(byte_filenames, labels_filenames)))
    
    # Verify filenames match
    # assert np.array_equal(cfg_filenames, ast_filenames), "Filenames do not match between CFG and AST"
    # assert np.array_equal(cfg_filenames, byte_filenames), "Filenames do not match between CFG and Bytecode"
    # assert np.array_equal(cfg_filenames, labels_filenames), "Filenames do not match between CFG and Labels"
    
    # Reorder embeddings according to the sorted labels filenames
    ast_data_sorted = reorder_embeddings(ast_data['embeddings'], ast_filenames, final_filenames)
    byte_data_sorted = reorder_embeddings(byte_data['embeddings'], byte_filenames, final_filenames)
    cfg_data_sorted = reorder_embeddings(cfg_data['embeddings'], cfg_filenames, final_filenames)
    labels_data_sorted = reorder_embeddings(labels['Label'].to_numpy(), labels_filenames, final_filenames)
    
    return ast_data_sorted, byte_data_sorted, cfg_data_sorted, labels_data_sorted, final_filenames

def reorder_embeddings(embeddings: np.ndarray, source_filenames: list, target_filenames: list) -> np.ndarray:
    """Reorder the embeddings to match the order of target_filenames."""
    # Create a dictionary to map source filenames to their corresponding embeddings
    filename_to_embedding = {source_filenames[i]: embeddings[i] for i in range(len(source_filenames))}
    
    # Reorder embeddings based on target_filenames order
    reordered_embeddings = np.array([filename_to_embedding[filename] for filename in target_filenames])
    
    return reordered_embeddings

def main():
    # Paths
    cfg_path = 'embeddings_cfg/cfg_embeddings.npz'
    ast_path = 'embeddings_ast/ast_embeddings.npz'
    byte_path = 'embeddings_bytecode/bytecode_embeddings.npz'
    labels_path = 'dataset/labels.csv'
    output_dir = Path('concatenated_embeddings')
    output_dir.mkdir(exist_ok=True)
    
    # Load embeddings
    print("Loading embeddings...")
    cfg_emb, ast_emb, byte_emb, labels, filenames = load_embeddings(cfg_path, ast_path, byte_path, labels_path)
    
    # Check shapes before concatenation
    print(f"CFG Embedding Shape: {cfg_emb.shape}")
    print(f"AST Embedding Shape: {ast_emb.shape}")
    print(f"Bytecode Embedding Shape: {byte_emb.shape}")
    
    # Ensure all embeddings have the same number of samples
    assert cfg_emb.shape[0] == ast_emb.shape[0] == byte_emb.shape[0], "Sample count mismatch between embeddings"
    
    # Concatenate embeddings horizontally (axis=1)
    combined_emb = np.concatenate((cfg_emb, ast_emb, byte_emb), axis=1)
    
    # Save concatenated embeddings
    output_path = output_dir / 'concatenated_embeddings.npz'
    np.savez(output_path,
             embeddings=combined_emb,
             labels=labels,
             filenames=filenames)
    
    print(f"Concatenated embeddings saved to {output_path}")
    print(f"Combined embedding shape: {combined_emb.shape}")

if __name__ == '__main__':
    main()
