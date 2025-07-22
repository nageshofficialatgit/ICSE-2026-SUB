import os
import json
import torch
import networkx as nx
from slither.slither import Slither
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel

# Initialize CodeBERT
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')
model.eval()

def codebert_embedding(text):
    """Get CodeBERT embedding for a code snippet or variable name."""
    with torch.no_grad():
        text = str(text).strip()
        if not text:
            return np.zeros(768)
        inputs = tokenizer(text, padding=True, truncation=True, max_length=32, return_tensors="pt")
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()  # [CLS] token
        return emb

def extract_dfg_from_contract(contract_path):
    slither = Slither(contract_path)
    dfg = nx.DiGraph()
    node_info = {}  # Map node name to code snippet or variable name
    for contract in slither.contracts:
        for function in contract.functions:
            # Add nodes for all variables read/written in the function
            for var in function.variables_read + function.variables_written:
                node_name = f"{function.full_name}:{var.name}"
                dfg.add_node(node_name)
                node_info[node_name] = var.name  # Use variable name as code snippet

            # Add edges for data dependencies
            for node in function.nodes:
                # For each variable written in this node, add edges from variables read
                for var_written in node.variables_written:
                    for var_read in node.variables_read:
                        src = f"{function.full_name}:{var_read.name}"
                        dst = f"{function.full_name}:{var_written.name}"
                        dfg.add_edge(src, dst)
    return dfg, node_info

def assign_features(G, node_info):
    for node in G.nodes():
        code_snippet = node_info.get(node, node)
        G.nodes[node]['x'] = codebert_embedding(code_snippet)
    return G

def process_and_save(contract_path, contract_address, output_dir):
    dfg, node_info = extract_dfg_from_contract(contract_path)
    dfg = assign_features(dfg, node_info)

    # Prepare edge list and node features
    node_list = list(dfg.nodes)
    node_idx_map = {n: i for i, n in enumerate(node_list)}
    edges = [(node_idx_map[src], node_idx_map[dst]) for src, dst in dfg.edges]
    node_features = np.stack([dfg.nodes[n]['x'] for n in node_list]) if node_list else np.zeros((0, 768))

    # Save as torch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)

    # Save .pt file
    torch.save({
        'edge_index': edge_index,
        'node_features': node_features_tensor,
        'num_nodes': node_features_tensor.shape[0],
        'node_names': node_list
    }, os.path.join(output_dir, f"{contract_address}_dfg_processed.pt"))

    # Save metadata as .json
    meta = {
        'num_nodes': node_features_tensor.shape[0],
        'num_edges': edge_index.shape[1],
        'feature_dim': node_features_tensor.shape[1],
        'filename': contract_address
    }
    with open(os.path.join(output_dir, f"{contract_address}_dfg_processed_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    contract_dir = "./contracts"  # Change as needed
    output_dir = "dfg_processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Recursively find all .sol files
    for sol_file in Path(contract_dir).rglob("*.sol"):
        path = str(sol_file)
        address = sol_file.stem
        print(f"Processing {path} ...")
        try:
            process_and_save(path, address, output_dir)
            print(f"Processed {path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

if __name__ == "__main__":
    main() 