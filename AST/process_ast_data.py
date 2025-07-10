import json
import os
import torch
from torch_geometric.data import Data


def parse_ast_to_graph(ast_json):
    """
    Parse a Solidity AST JSON from solc into a graph for PyTorch Geometric.

    Args:
        ast_json (dict): The AST JSON.

    Returns:
        Data: A PyTorch Geometric Data object containing the graph.
    """
    nodes = []  # Store nodes
    edges = []  # Store edges
    node_features = []  # Store node features
    node_id_map = {}  # Map to assign unique IDs to nodes
    node_id_counter = 0  # Counter to generate node IDs

    def add_node(node, parent_id=None):
        """
        Recursively process an AST node and its children.
        """
        nonlocal node_id_counter

        # Assign a unique ID to this node
        node_id = node_id_counter
        node_id_map[id(node)] = node_id
        node_id_counter += 1

        # Add the node type and name as features (use hash to encode)
        node_type = node.get("nodeType", "Unknown")
        node_name = node.get("name", "Unnamed")
        feature = [hash(node_type) % 1000, hash(node_name) % 1000]  # Combine hash of type and name for features
        nodes.append(node_id)
        node_features.append(feature)

        # Add an edge to the parent node (if it exists)
        if parent_id is not None:
            edges.append((parent_id, node_id))

        # Process child nodes: parameters, statements, body, etc.
        children_keys = ["nodes", "parameters", "statements", "arguments", "eventCall", "body"]
        for key in children_keys:
            if key in node:
                child = node[key]
                if isinstance(child, list):
                    for child_node in child:
                        add_node(child_node, node_id)
                elif isinstance(child, dict):
                    add_node(child, node_id)

    # Start processing from the root node
    add_node(ast_json)

    # Convert edges and node features to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    return x, edge_index


def preprocess_ast(json_file, filename, output_dir):
    """
    Preprocess a Solidity AST JSON file into a PyTorch Geometric graph.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    try:
        with open(json_file, "r") as file:
            ast_json = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {json_file}")
        return None
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return None

    graph_data = parse_ast_to_graph(ast_json)

    output_path = os.path.join(output_dir, f"{filename[:-5]}_processed")
    
    # Save tensors
    torch.save({
        'edge_index': graph_data[1],
        'node_features': graph_data[0],
        'num_nodes': len(graph_data[0])
    }, str(output_path) + '.pt')

    return graph_data


# Example usage
if __name__ == "__main__":
    # Replace with the path to your AST JSON file
    input_dir = "AST/ast_outputs"
    output_dir = "AST/processed_ast_data"

    # Preprocess AST into a graph
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            json_file = os.path.join(input_dir, file)
            try:
                graph = preprocess_ast(json_file, file, output_dir)
                if graph:
                    print(f"Processed {file} successfully.")
            except Exception as e:
                print(f"Error processing {file}: {e}")
