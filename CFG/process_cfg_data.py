import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import RobertaTokenizer, RobertaModel

class CFGFeatureExtractor:
    def __init__(self):
        # Initialize CodeBERT tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.model.eval()  # Set to evaluation mode
        
        # Complete list of Slither NodeTypes
        self.node_types = {
            'ENTRY_POINT': 0,
            'EXIT_POINT': 1,
            'EXPRESSION': 2,
            'RETURN': 3,
            'IF': 4,
            'IFLOOP': 5,
            'THROW': 6,
            'BREAK': 7,
            'CONTINUE': 8,
            'PLACEHOLDER': 9,
            'TRY': 10,
            'CATCH': 11,
            'ASSEMBLY': 12,
            'VARIABLE': 13,
            'VARIABLE_INIT': 14,
            'NEW_VARIABLE': 15,
            'NEW_CONTRACT': 16,
            'INLINE_ASM': 17,
            'MODIFIER': 18,
            'MODIFIER_CALL': 19,
            'EVENT_CALL': 20,
            'SOLIDITY_CALL': 21,
            'MEMBER': 22,
            'TUPLE': 23,
            'CONDITIONAL': 24,
            'OPERATION': 25,
            'TYPE_CONVERSION': 26,
            'INDEX_ACCESS': 27,
            'LITERAL': 28,
            'IDENTIFIER': 29,
            'ASSIGNMENT': 30,
            'BINARY': 31,
            'UNARY': 32,
            'CALL': 33,
            'LOCAL_VARIABLE_INIT': 34,
            'MEMBER_ACCESS': 35,
            'REQUIRE': 36,
            'ASSERT': 37,
            'REVERT': 38,
            'STARTLOOP': 39,
            'ENDLOOP': 40
        }
        
    def get_code_embedding(self, code: str) -> np.ndarray:
        """Generate CodeBERT embedding for a code snippet"""
        try:
            with torch.no_grad():
                # Clean the code string
                code = str(code).strip()
                if not code:
                    return np.zeros(768)  # Return zero vector for empty code
                
                # Tokenize and encode
                inputs = self.tokenizer(code, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt")
                
                # Get embeddings
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as code representation
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                return embeddings[0]
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return np.zeros(768)

    def create_node_features(self, node: Dict) -> np.ndarray:
        """Create feature vector for a single node"""
        try:
            # One-hot encoding for node type
            node_type_one_hot = np.zeros(len(self.node_types))
            
            # Extract the node type from the full type string
            node_type = node.get('Type', '')
            if not node_type:
                print(f"Warning: Empty node type for node {node.get('ID', 'unknown')}")
                return np.zeros(len(self.node_types) + 768)
            
            node_type = node_type.upper()
            if 'NODETYPE.' in node_type:
                node_type = node_type.split('NODETYPE.')[-1]
            
            # Handle node type matching
            matched_type = None
            for type_key in self.node_types.keys():
                if type_key in node_type:
                    matched_type = type_key
                    break
            
            if matched_type:
                node_type_one_hot[self.node_types[matched_type]] = 1
            else:
                print(f"Warning: Unknown node type: {node_type} for node {node.get('ID', 'unknown')}")
            
            # Get CodeBERT embedding for the expression
            expression = node.get('Expression', '')
            expression_embedding = self.get_code_embedding(expression)
            
            # Concatenate features
            return np.concatenate([node_type_one_hot, expression_embedding])
        
        except Exception as e:
            print(f"Error in create_node_features for node {node.get('ID', 'unknown')}: {e}")
            return np.zeros(len(self.node_types) + 768)

    def process_cfg(self, cfg_data: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Process CFG data to create edge list and node features"""
        nodes = []
        edges = []
        current_node = None
        
        # Split the data into lines and process each line
        lines = cfg_data.split('\n')
        in_nodes_section = False
        in_edges_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check section headers
            if line == "Nodes:":
                in_nodes_section = True
                in_edges_section = False
                continue
            elif line == "Edges:":
                in_nodes_section = False
                in_edges_section = True
                continue
            
            # Process nodes
            if in_nodes_section:
                if line.startswith("ID:"):
                    if current_node:
                        nodes.append(current_node)
                    current_node = {'ID': int(line.split("ID:")[-1].strip())}
                elif line.startswith("Type:"):
                    if current_node:
                        current_node['Type'] = line.split("Type:")[-1].strip()
                elif line.startswith("Expression:"):
                    if current_node:
                        current_node['Expression'] = line.split("Expression:")[-1].strip()
            
            # Process edges
            if in_edges_section and "->" in line:
                # Remove any leading spaces or other characters
                edge_str = line.strip().split("->")
                try:
                    source = int(edge_str[0].strip())
                    target = int(edge_str[1].strip())
                    edges.append((source, target))
                except ValueError as e:
                    print(f"Error parsing edge: {line} - {e}")
        
        # Add the last node if exists
        if current_node:
            nodes.append(current_node)
        
        # Sort nodes by ID to ensure consistent ordering
        nodes.sort(key=lambda x: x['ID'])
        
        # Create node features
        node_features = []
        for node in nodes:
            try:
                features = self.create_node_features(node)
                node_features.append(features)
            except Exception as e:
                print(f"Error creating features for node {node['ID']}: {e}")
                # Add zero features for failed nodes to maintain consistency
                zero_features = np.zeros(len(self.node_types) + 768)  # node_types + CodeBERT dim
                node_features.append(zero_features)
        
        if not nodes:
            print("Warning: No nodes found in CFG data")
            return [], np.array([])
        
        if not edges:
            print("Warning: No edges found in CFG data")
        
        return edges, np.array(node_features)

class CFGDataProcessor:
    def __init__(self, cfg_dir: str, output_dir: str):
        self.cfg_dir = Path(cfg_dir)
        self.output_dir = Path(output_dir)
        self.extractor = CFGFeatureExtractor()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def read_cfg_file(self, file_path: Path) -> str:
        """Read CFG file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def process_single_cfg(self, file_path: Path) -> Dict:
        """Process a single CFG file and return its features"""
        cfg_data = self.read_cfg_file(file_path)
        edges, node_features = self.extractor.process_cfg(cfg_data)
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        return {
            'edge_index': edge_index,
            'node_features': node_features,
            'num_nodes': node_features.shape[0],
            'filename': file_path.stem
        }
    
    def save_processed_data(self, data: Dict, filename: str):
        """Save processed data in PyTorch format"""
        output_path = self.output_dir / f"{filename}_processed"
        
        # Save tensors
        torch.save({
            'edge_index': data['edge_index'],
            'node_features': data['node_features'],
            'num_nodes': data['num_nodes']
        }, str(output_path) + '.pt')
        
        # Save metadata
        metadata = {
            'num_nodes': data['num_nodes'],
            'num_edges': data['edge_index'].shape[1],
            'feature_dim': data['node_features'].shape[1],
            'filename': data['filename']
        }
        
        with open(str(output_path) + '_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_all_cfgs(self):
        """Process all CFG files in the directory"""
        cfg_files = list(self.cfg_dir.glob('*_cfg.txt'))
        processed_data = []
        
        print(f"Found {len(cfg_files)} CFG files to process")
        
        for file_path in cfg_files:
            print(f"Processing {file_path.name}...")
            try:
                data = self.process_single_cfg(file_path)
                processed_data.append(data)
                self.save_processed_data(data, file_path.stem)
                print(f"Successfully processed {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
        
        # Save dataset statistics
        stats = {
            'total_graphs': len(processed_data),
            'avg_nodes': np.mean([d['num_nodes'] for d in processed_data]),
            'avg_edges': np.mean([d['edge_index'].shape[1] for d in processed_data]),
            'feature_dim': processed_data[0]['node_features'].shape[1] if processed_data else 0
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return processed_data

def main():
    # Define directories with absolute paths
    current_dir = Path(__file__).parent.absolute()
    cfg_dir = current_dir / "cfg_outputs"
    output_dir = current_dir / "processed_cfg_data"
    
    # Verify directories exist
    if not cfg_dir.exists():
        print(f"Error: CFG directory not found at {cfg_dir}")
        print("Current directory structure:")
        print_directory_structure(current_dir)
        return
    
    # Check if there are any CFG files
    cfg_files = list(cfg_dir.glob('*_cfg.txt'))
    if not cfg_files:
        print(f"Error: No CFG files found in {cfg_dir}")
        print("Files in cfg_outputs directory:")
        if cfg_dir.exists():
            print([f.name for f in cfg_dir.iterdir()])
        return
    
    # Create processor and process all CFGs
    try:
        processor = CFGDataProcessor(str(cfg_dir), str(output_dir))
        processed_data = processor.process_all_cfgs()
        
        print("\nProcessing complete!")
        print(f"Processed {len(processed_data)} CFG files")
        print(f"Output saved to: {output_dir}")
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

def print_directory_structure(startpath):
    """Helper function to print directory structure"""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(str(startpath), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    main() 