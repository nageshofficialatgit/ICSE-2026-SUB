

import os
import json
import torch
import networkx as nx
from slither.slither import Slither
import numpy as np
from pathlib import Path

def extract_cfg_from_contract(contract_path):
    slither = Slither(contract_path)
    cfg = {}
    for contract in slither.contracts:
        for function in contract.functions:
            fn_name = function.full_name
            edges = [(n.source_mapping.start, d.source_mapping.start)
                     for n in function.nodes for d in n.sons]
            cfg[fn_name] = edges
    return cfg

def extract_call_graph(contract_path):
    slither = Slither(contract_path)
    call_edges = []
    for contract in slither.contracts:
        for function in contract.functions:
            caller = function.full_name
            for call in function.high_level_calls:
                callee = call.destination.full_name
                if callee:
                    call_edges.append((caller, callee))
    return call_edges

def build_ccfg(cfg_dict, call_edges):
    G = nx.DiGraph()
    for fn, edges in cfg_dict.items():
        for src, dst in edges:
            G.add_edge(f"{fn}:{src}", f"{fn}:{dst}")  # CFG edges
    for src_fn, dst_fn in call_edges:
        G.add_edge(src_fn, dst_fn)  # Call Graph edges
    return G

def assign_features(G, feature_dim=16):
    for node in G.nodes():
        G.nodes[node]['x'] = np.random.rand(feature_dim)  # Placeholder features
    return G

def process_and_save(contract_path, contract_address, output_dir, feature_dim=16):
    cfg = extract_cfg_from_contract(contract_path)
    call_graph = extract_call_graph(contract_path)
    ccfg = build_ccfg(cfg, call_graph)
    ccfg = assign_features(ccfg, feature_dim=feature_dim)

    # Prepare edge list and node features
    node_list = list(ccfg.nodes)
    node_idx_map = {n: i for i, n in enumerate(node_list)}
    edges = [(node_idx_map[src], node_idx_map[dst]) for src, dst in ccfg.edges]
    node_features = np.stack([ccfg.nodes[n]['x'] for n in node_list])

    # Save as torch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)

    # Save .pt file
    torch.save({
        'edge_index': edge_index,
        'node_features': node_features_tensor,
        'num_nodes': node_features_tensor.shape[0],
        'node_names': node_list
    }, os.path.join(output_dir, f"{contract_address}_ccfg_processed.pt"))

    # Save metadata as .json
    meta = {
        'num_nodes': node_features_tensor.shape[0],
        'num_edges': edge_index.shape[1],
        'feature_dim': node_features_tensor.shape[1],
        'filename': contract_address
    }
    with open(os.path.join(output_dir, f"{contract_address}_ccfg_processed_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    contract_dir = "./contracts"  # Change as needed
    output_dir = "ccfg_processed_data"
    os.makedirs(output_dir, exist_ok=True)
    feature_dim = 16

    # Recursively find all .sol files
    for sol_file in Path(contract_dir).rglob("*.sol"):
        path = str(sol_file)
        address = sol_file.stem
        print(f"Processing {path} ...")
        try:
            process_and_save(path, address, output_dir, feature_dim=feature_dim)
            print(f"Processed {path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

if __name__ == "__main__":
    main()