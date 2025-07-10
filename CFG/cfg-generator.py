import os
from slither import Slither
from slither.core.cfg.node import NodeType
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import re

def get_solc_version(file_path):
    """Extract solc version from pragma statement"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('pragma solidity'):
                    match = re.search(r'(\d+\.\d+\.\d+)', line)
                    if match:
                        return match.group(1)
        return '0.4.26'  # Default version
    except Exception as e:
        print(f"Version detection error: {str(e)}")
        return '0.4.26'

def check_install_solc(version):
    """Check and install solc version if needed"""
    try:
        # Check installed versions
        result = subprocess.run(['solc-select', 'versions'], 
                              capture_output=True, text=True)
        if version in result.stdout:
            return True
        
        # Install missing version
        print(f"Installing solc {version}...")
        subprocess.run(['solc-select', 'install', version], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install solc {version}: {e.stderr}")
        return False

def generate_contract_cfg(sol_file_path):
    try:
        # Version management
        required_version = get_solc_version(sol_file_path)
        if not check_install_solc(required_version):
            print(f"Skipping {sol_file_path} - required solc {required_version} unavailable")
            return None, None
        
        # Activate required version
        subprocess.run(['solc-select', 'use', required_version], check=True)
        
        # Original Slither processing
        slither = Slither(sol_file_path)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Store contract information
        contract_info = {}
        
        # Process each contract in the Solidity file
        for contract in slither.contracts:
            print(f"Generating CFG for contract: {contract.name}")
            contract_info[contract.name] = {
                'functions': {},
                'relationships': []
            }
            
            # Process each function in the contract
            for function in contract.functions:
                if function.nodes:
                    # Store function information
                    contract_info[contract.name]['functions'][function.name] = {
                        'nodes': [],
                        'edges': []
                    }
                    
                    # Add nodes and edges from the function's CFG
                    for node in function.nodes:
                        # Add node with label
                        node_label = f"{function.name}:\n{str(node.type)}\n{node.expression if node.expression else ''}"
                        G.add_node(node.node_id, label=node_label)
                        contract_info[contract.name]['functions'][function.name]['nodes'].append({
                            'id': node.node_id,
                            'type': str(node.type),
                            'expression': str(node.expression) if node.expression else ''
                        })
                        
                        # Add edges to successor nodes
                        for successor in node.sons:
                            G.add_edge(node.node_id, successor.node_id)
                            contract_info[contract.name]['functions'][function.name]['edges'].append({
                                'from': node.node_id,
                                'to': successor.node_id
                            })
        
        return G, contract_info
    
    except subprocess.CalledProcessError as e:
        print(f"Version switch failed for {sol_file_path}: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"CFG generation error: {str(e)}")
        return None, None

def save_cfg_data(contract_info, output_path):
    """Save CFG data to a text file"""
    try:
        with open(output_path, 'w') as f:
            for contract_name, contract_data in contract_info.items():
                f.write(f"Contract: {contract_name}\n")
                f.write("=" * 50 + "\n\n")
                
                for func_name, func_data in contract_data['functions'].items():
                    f.write(f"Function: {func_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    f.write("Nodes:\n")
                    for node in func_data['nodes']:
                        f.write(f"  ID: {node['id']}\n")
                        f.write(f"  Type: {node['type']}\n")
                        f.write(f"  Expression: {node['expression']}\n")
                        f.write("\n")
                    
                    f.write("Edges:\n")
                    for edge in func_data['edges']:
                        f.write(f"  {edge['from']} -> {edge['to']}\n")
                    f.write("\n")
                f.write("\n")
        
        print(f"CFG data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving CFG data: {str(e)}")

def visualize_cfg(graph, output_path):
    try:
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)
        
        plt.title("Contract Control Flow Graph")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        print(f"CFG visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error visualizing CFG: {str(e)}")

def main():
    # Path to the dataset folder using Windows-style paths
    dataset_path = 'dataset\\multivul_multiclass_test'
    
    # Create output directories
    output_dir = 'CFG\\cfg_outputs'
    image_dir = 'CFG\\cfg_images'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Process each .sol file in the dataset folder
    for filename in os.listdir(dataset_path):
        if filename.endswith('.sol'):
            sol_file_path = os.path.join(dataset_path, filename)
            
            # Skip files that can't be processed
            if not os.path.isfile(sol_file_path):
                print(f"Skipping invalid file: {filename}")
                continue
            
            cfg_output_path = os.path.join(output_dir, f"{filename[:-4]}_cfg.txt")
            image_output_path = os.path.join(image_dir, f"{filename[:-4]}_cfg.png")
            
            print(f"Processing: {filename}")
            
            # Generate CFG
            cfg_graph, contract_info = generate_contract_cfg(sol_file_path)
            
            if cfg_graph and contract_info:
                # Save CFG data
                save_cfg_data(contract_info, cfg_output_path)
                # Visualize and save CFG
                visualize_cfg(cfg_graph, image_output_path)
            else:
                print(f"Failed to generate CFG for {filename}")

if __name__ == "__main__":
    main()