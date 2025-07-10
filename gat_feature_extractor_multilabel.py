import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import optuna
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

source = 'CFG'

class GraphDataset(Dataset):
	def __init__(self, processed_dir: str, label_csv: str, transform=None):
		self._processed_dir = Path(processed_dir)
		labels_df = pd.read_csv(label_csv)
		# Build a list of (file, label) tuples from the CSV, using only the part before '.sol'
		file_label_pairs = []
		for _, row in labels_df.iterrows():
			file_id_full = str(row['filename'])
			file_id = file_id_full.split('.sol')[0]  # only part before .sol
			label = row[1:].to_numpy().astype(np.float32)
			file_label_pairs.append((file_id, label))
		# Build a set for fast lookup
		file_id_set = set(file_id for file_id, _ in file_label_pairs)
		# Map from file id to processed file path, using only the part before '_ast_processed.pt' or '_cfg_processed.pt'
		all_files = list(self._processed_dir.glob('*_processed.pt'))
		file_id_to_path = {}
		for f in all_files:
			fname = f.as_posix().split('/')[-1]
			if '_ast_processed' in fname:
				idx = fname.split('_ast_processed')[0]
			elif '_cfg_processed' in fname:
				idx = fname.split('_cfg_processed')[0]
			else:
				idx = fname.split('_processed')[0]
			if idx in file_id_set:
				file_id_to_path[idx] = f
		# Only keep files that are in the CSV, in the order of the CSV
		self.file_paths = []
		self.labels = []
		for file_id, label in file_label_pairs:
			if file_id in file_id_to_path:
				self.file_paths.append(file_id_to_path[file_id])
				self.labels.append(label)
		self.labels = torch.tensor(self.labels, dtype=torch.float32)
		if len(self.labels) == 0:
			raise ValueError("No matching files found between processed_dir and CSV. Check filename conventions and CSV content.")
		self.file_cache = {}
		super().__init__(root=processed_dir, transform=transform)
		
	@property
	def processed_dir(self) -> Path:
		return self._processed_dir
	
	def len(self):
		return len(self.file_paths)
	
	def get(self, idx):
		# Load processed data
		label = self.labels[idx]
		
		if idx in self.file_cache:
			return self.file_cache[idx], label
		
		file_path = self.file_paths[idx]
		data_dict = torch.load(file_path)
		
		# Validation check
		num_nodes = data_dict['node_features'].size(0)
		invalid_indices = (data_dict['edge_index'] >= num_nodes).any(dim=0)
		if invalid_indices.any():
			print(f"Invalid edge indices in file: {file_path.name}")
			print(f"Node count: {num_nodes}, Max edge index: {data_dict['edge_index'].max().item()}")
			# Clamp invalid indices to valid range
			data_dict['edge_index'] = torch.clamp(data_dict['edge_index'], 0, num_nodes-1)

		# Create PyG Data object
		data = Data(
			x=data_dict['node_features'],
			edge_index=data_dict['edge_index']
		)

		self.file_cache[idx] = data
		
		return data, label

	# Required properties for PyG Dataset
	@property
	def raw_file_names(self):
		return []

	@property
	def processed_file_names(self):
		# Now file_paths will always exist when this is called
		return [f.name for f in self.file_paths]

class GAT(nn.Module):
	def __init__(self, 
				 in_channels: int,
				 hidden_channels: int,
				 num_layers: int = 3,
				 heads: int = 4,
				 dropout: float = 0.2,
				 pool_ratio: float = 0.5):
		super(GAT, self).__init__()
		
		self.num_layers = num_layers
		self.convs = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		
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
		# Validate edge indices before processing
		if x.size(0) > 0 and edge_index.numel() > 0:
			max_index = edge_index.max().item()
			if max_index >= x.size(0):
				print(f"Warning: Invalid edge index {max_index} in graph with {x.size(0)} nodes")
				# Clamp invalid indices
				edge_index = torch.clamp(edge_index, 0, x.size(0)-1)

		# Handle empty graphs
		if edge_index.numel() == 0:
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
	
class Classifier(nn.Module):
	def __init__(self, input_dim, num_classes):
		super(Classifier, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.Sigmoid(),
			nn.Dropout(0.3),
			nn.Linear(256, 128),
			nn.Tanh(),
			nn.Dropout(0.2),
			nn.Linear(128, num_classes)
		)
		
	def forward(self, x: torch.Tensor):
		return self.layers(x)

class EmbeddingGenerator:
	def __init__(self, 
				 model_params: dict,
				 processed_dir: str,
				 label_csv: str,
				 output_dir: str):
		self.processed_dir = Path(processed_dir)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.criterion = nn.BCEWithLogitsLoss()
		
		# Load dataset
		self.dataset = GraphDataset(processed_dir, label_csv)
		num_classes = self.dataset.labels.shape[1]

		print("self.dataset", self.dataset.get(0))
		
		# Initialize model
		self.model = GAT(
			in_channels=self.dataset.get(0)[0].num_features,
			**model_params
		)

		self.classifier = Classifier(model_params['hidden_channels'], num_classes)
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)
		self.classifier = self.classifier.to(self.device)
	
	def train_model(self, 
				   num_epochs: int,
				   batch_size: int,
				   learning_rate: float):
		"""Train the GAT model using contrastive learning"""
		loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
		optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
		
		self.model.train()
		losses = []
		
		for epoch in range(num_epochs):
			epoch_loss = 0
			correct = 0
			total = 0
			for batch, label in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
				batch = batch.to(self.device)
				label = label.to(self.device)
				optimizer.zero_grad()
				
				# Get embeddings
				embeddings = self.model(batch.x, batch.edge_index, batch.batch)
				pred = self.classifier(embeddings)
				correct += ((F.sigmoid(pred) > 0.5).float() == label).all(dim = 1).float().sum().item()
				total += label.numel()
				loss = self.criterion(pred, label)
				loss.backward()
				optimizer.step()
				
				epoch_loss += loss.item()
			
			avg_loss = epoch_loss / len(loader)
			losses.append(avg_loss)
			print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {(correct / total):.4f}')
		
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
			for idx, (data, label) in enumerate(loader):
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
	
	def save_embeddings(self, embeddings: np.ndarray, filenames: 'list[str]'):
		"""Save embeddings to file"""
		output_path = self.output_dir / (source.lower() + '_embeddings.npz')
		np.savez(output_path, 
				 embeddings=embeddings,
				 filenames=filenames)
		print(f"Embeddings saved to {output_path}")
	
	def plot_training_loss(self, losses: 'list[float]'):
		"""Plot training loss curve"""
		plt.figure(figsize=(10, 6))
		plt.plot(losses)
		plt.title('Training Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.savefig(self.output_dir / 'training_loss.png')
		plt.close()
	
	def visualize_embeddings(self, embeddings: np.ndarray, filenames: 'list[str]'):
		"""Visualize embeddings using t-SNE, coloring by unique label combination"""
		# Reduce dimensionality using t-SNE
		tsne = TSNE(n_components=2, random_state=42)
		embeddings_2d = tsne.fit_transform(embeddings)

		# Get the labels for the current dataset (in the same order as embeddings)
		labels = self.labels.cpu().numpy() if torch.is_tensor(self.labels) else np.array(self.labels)
		# Convert each label vector to a tuple for unique combination mapping
		label_tuples = [tuple(l) for l in labels]
		unique_combos = sorted(set(label_tuples))
		# Assign a color to each unique label combination
		color_list = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
		color_map = {combo: color_list[i % len(color_list)] for i, combo in enumerate(unique_combos)}
		# Print the mapping
		print("t-SNE color mapping (label combination -> color):")
		for combo, color in color_map.items():
			print(f"{combo}: {color}")
		# Assign color to each point
		point_colors = [color_map[combo] for combo in label_tuples]

		# Plot
		plt.figure(figsize=(12, 8))
		scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, alpha=0.7)
		# Add legend for each unique label combination
		handles = [mpatches.Patch(color=color_map[combo], label=str(combo)) for combo in unique_combos]
		plt.legend(handles=handles, title="Label Combination", bbox_to_anchor=(1.05, 1), loc='upper left')
		# Add labels for some points
		for i, filename in enumerate(filenames):
			if i % max(1, len(filenames) // 20) == 0:  # Label every nth point
				plt.annotate(filename, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
		plt.title('t-SNE visualization of embeddings (colored by label combination)')
		plt.tight_layout()
		plt.savefig(self.output_dir / 'embeddings_tsne.png')
		plt.close()

def optimize_hyperparameters(processed_dir: str, label_csv: str, output_dir: str, n_trials: int = 16):
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
			label_csv=label_csv,
			output_dir=output_dir+'/implement/embeddings'
		)
		
		losses = generator.train_model(**training_params)
		return losses[-1]  # Return final loss
	
	study = optuna.create_study(direction='minimize')
	study.optimize(objective, n_trials=n_trials)
	
	return study.best_params

def main():
	global source
	source = 'CFG'
	#source = 'AST'
	processed_dir = f"./{source}/processed_{source.lower()}_data"
	label_csv = 'dataset/multivul_multiclass_test/Labels.csv'
	output_dir = "./embeddings_" + source.lower()
	
	# # Optimize hyperparameters
	# print("Optimizing hyperparameters...")
	# best_params = optimize_hyperparameters(processed_dir, label_csv, output_dir, n_trials=16)
	# print("Best hyperparameters:", best_params)
	
	# # Train model with best parameters
	# model_params = {
	#     'hidden_channels': best_params['hidden_channels'],
	#     'num_layers': best_params['num_layers'],
	#     'heads': best_params['heads'],
	#     'dropout': best_params['dropout']
	# }
	
	# training_params = {
	#     'num_epochs': 50,
	#     'batch_size': best_params['batch_size'],
	#     'learning_rate': best_params['learning_rate']
	# }

	model_params = {
		'hidden_channels': 128,
		'num_layers': 8,
		'heads': 8,
		'dropout': 0.25
	}
	
	training_params = {
		'num_epochs': 30,
		'batch_size': 32,
		'learning_rate': 5e-4
	}
	
	# Generate embeddings
	generator = EmbeddingGenerator(
		model_params=model_params,
		processed_dir=processed_dir,
		label_csv=label_csv,
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