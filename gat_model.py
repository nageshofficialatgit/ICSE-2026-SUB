import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
import json
from typing import Dict, List
import optuna
from tqdm import tqdm
import os

source = 'CFG'

class GraphDataset(Dataset):
    def __init__(self, processed_dir: str, transform=None):
        self._processed_dir = Path(processed_dir)
        
        # First load the CSV to get the filenames we need to look for
        labels_df = pd.read_csv(os.path.join('dataset', 'pure_vul_test', 'Labels.csv'))
        # Get base names without .sol extension
        csv_filenames = [f.replace('.sol', '') for f in labels_df['File'].tolist()]
        print(f"Number of files in CSV: {len(csv_filenames)}")
        print(f"First few CSV filenames: {csv_filenames[:5]}")
        
        # Get all processed files
        all_pt_files = list(self._processed_dir.glob('*_processed.pt'))
        print(f"Number of .pt files found: {len(all_pt_files)}")
        print(f"First few .pt files: {[f.name for f in all_pt_files[:5]]}")
        
        # Find which CSV files have corresponding .pt files
        self.file_paths = []
        # Not using labels for contrastive learning, but can be loaded for other purposes
        self.labels = []
        
        for csv_base_name in csv_filenames:
            # Find all .pt files that start with this base name
            matching_files = [f for f in all_pt_files if f.stem.startswith(csv_base_name)]
            if matching_files:
                # Take the first matching file
                self.file_paths.append(matching_files[0])
                # Get the label for this file (add .sol back to match CSV)
                label = labels_df[labels_df['File'] == csv_base_name + '.sol']['Label'].iloc[0]
                self.labels.append(label)
                print(f"Found match: {csv_base_name} -> {matching_files[0].name}")
        
        print(f"\nSummary:")
        print(f"Number of matching files found: {len(self.file_paths)}")
        print(f"Number of labels: {len(self.labels)}")
        
        super().__init__(root=processed_dir, transform=transform)
        
    @property
    def processed_dir(self) -> Path:
        return self._processed_dir
    
    def len(self):
        return len(self.file_paths)
    
    def get(self, idx):
        # Load processed data
        data_dict = torch.load(self.file_paths[idx])
        
        # Create PyG Data object
        data = Data(
            x=data_dict['node_features'],
            edge_index=data_dict['edge_index']
        )
        
        return data

    # Required properties for PyG Dataset
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Now file_paths will always exist when this is called
        return [f.name for f in self.file_paths]

class GAT(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.2,
                 pool_ratio: float = 0.5):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, add_self_loops=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, add_self_loops=True)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, add_self_loops=True)
        )
        
        # Final embedding projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
    def forward(self, x, edge_index, batch):
        # Handle empty graphs
        if edge_index.numel() == 0:
            # Create self-loops for isolated nodes
            num_nodes = x.size(0)
            edge_index = torch.arange(num_nodes, device=x.device)
            edge_index = torch.stack([edge_index, edge_index], dim=0)
        
        # Initial features
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final GAT layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling (combine max and mean pooling)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_max, x_mean], dim=1)
        
        # Final projection
        return self.projection(x)

class EmbeddingGenerator:
    def __init__(self, 
                 model_params: Dict,
                 processed_dir: str,
                 output_dir: str):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = GraphDataset(processed_dir)

        print("self.dataset", self.dataset.get(0))
        
        # Initialize model
        self.model = GAT(
            in_channels=self.dataset.get(0).num_features,
            **model_params
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def train_model(self, 
                   num_epochs: int,
                   batch_size: int,
                   learning_rate: float):
        """Train the GAT model using contrastive learning"""
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Get embeddings
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Compute contrastive loss
                loss = self.contrastive_loss(embeddings)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        self.plot_training_loss(losses)
        return losses
    
    def contrastive_loss(self, embeddings: torch.Tensor, temperature: float = 0.5):
        """Compute contrastive loss for the embeddings"""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Mask out self-similarity
        mask = torch.eye(embeddings.shape[0], device=self.device)
        mask = 1 - mask
        
        # Compute loss
        similarity_matrix = similarity_matrix * mask
        positives = similarity_matrix.diag()
        negatives = similarity_matrix.sum(dim=1) - positives
        
        loss = -torch.log(torch.exp(positives / temperature) / 
                         torch.exp(negatives / temperature).sum(dim=0))
        return loss.mean()
    
    def generate_embeddings(self):
        """Generate embeddings for all graphs"""
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        embeddings = []
        filenames = []
        
        with torch.no_grad():
            for idx, data in enumerate(loader):
                data = data.to(self.device)
                
                # Skip empty graphs or add handling
                if data.edge_index.numel() == 0:
                    print(f"Warning: Graph {idx} has no edges. Adding self-loops.")
                
                embedding = self.model(data.x, data.edge_index, data.batch)
                embeddings.append(embedding.cpu().numpy())
                filenames.append(self.dataset.file_paths[idx].stem)
        
        embeddings = np.vstack(embeddings)
        self.save_embeddings(embeddings, filenames)
        self.visualize_embeddings(embeddings, filenames)
        return embeddings, filenames
    
    def save_embeddings(self, embeddings: np.ndarray, filenames: List[str]):
        """Save embeddings to file"""
        output_path = self.output_dir / (source.lower() + '_embeddings.npz')
        np.savez(output_path, 
                 embeddings=embeddings,
                 filenames=filenames)
        print(f"Embeddings saved to {output_path}")
    
    def plot_training_loss(self, losses: List[float]):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(self.output_dir / 'training_loss.png')
        plt.close()
    
    def visualize_embeddings(self, embeddings: np.ndarray, filenames: List[str]):
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
        plt.savefig(self.output_dir / 'embeddings_tsne.png')
        plt.close()

def optimize_hyperparameters(processed_dir: str, output_dir: str, n_trials: int = 4):
    """Optimize hyperparameters using Optuna"""
    def objective(trial):
        params = {
            'hidden_channels': trial.suggest_int('hidden_channels', 64, 128),
            'num_layers': trial.suggest_int('num_layers', 4, 8),
            'heads': trial.suggest_int('heads', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        }
        
        training_params = {
            'num_epochs': 20,
            'batch_size': trial.suggest_int('batch_size', 8, 32),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
        }
        
        generator = EmbeddingGenerator(
            model_params=params,
            processed_dir=processed_dir,
            output_dir=output_dir+'/implement/embeddings'
        )
        
        losses = generator.train_model(**training_params)
        return losses[-1]  # Return final loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def main():
    global source
   # source = 'CFG'
    source = 'AST'
    processed_dir = f"./{source}/processed_{source.lower()}_data"
    output_dir = "./embeddings_" + source.lower()
    
    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(processed_dir, output_dir, n_trials=16)
    print("Best hyperparameters:", best_params)
    
    # Train model with best parameters
    model_params = {
        'hidden_channels': best_params['hidden_channels'],
        'num_layers': best_params['num_layers'],
        'heads': best_params['heads'],
        'dropout': best_params['dropout']
    }
    
    training_params = {
        'num_epochs': 50,
        'batch_size': best_params['batch_size'],
        'learning_rate': best_params['learning_rate']
    }

    # model_params = {
    #     'hidden_channels': 128,
    #     'num_layers': 4,
    #     'heads': best_params['heads'],
    #     'dropout': best_params['dropout']
    # }
    
    # training_params = {
    #     'num_epochs': 100,
    #     'batch_size': best_params['batch_size'],
    #     'learning_rate': best_params['learning_rate']
    # }
    
    # Generate embeddings
    generator = EmbeddingGenerator(
        model_params=model_params,
        processed_dir=processed_dir,
        output_dir=output_dir
    )
    
    print("Training model...")
    generator.train_model(**training_params)
    
    print("Generating embeddings...")
    embeddings, filenames = generator.generate_embeddings()
    
    print("Process completed!")
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main() 