from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import optuna
from tqdm import tqdm
import os
import json
import pandas as pd

from combine_embeddings import MultiModalFusion, load_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLabelVulnerabilityDataset(Dataset):
    def __init__(self, ast_emb, byte_emb, cfg_emb, labels):
        self.ast_emb = torch.FloatTensor(ast_emb)
        self.byte_emb = torch.FloatTensor(byte_emb)
        self.cfg_emb = torch.FloatTensor(cfg_emb)
        self.labels = torch.FloatTensor(labels)  # Multi-hot encoded labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ast_emb[idx], self.byte_emb[idx], self.cfg_emb[idx], self.labels[idx]

class MemoryBank:
    def __init__(self, size, feature_dim):
        self.size = size
        self.features = torch.randn(size, feature_dim)
        self.features = F.normalize(self.features, dim=1)
        self.ptr = 0
        
    def update(self, new_features, ptr=None):
        if ptr is None:
            ptr = self.ptr
        batch_size = new_features.size(0)
        self.features[ptr:ptr + batch_size] = new_features
        self.ptr = (ptr + batch_size) % self.size
        return ptr
        
    def get_all_features(self):
        return self.features

class EnhancedContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return features, F.normalize(projections, dim=1)

class SimpleNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, embed_dim)
        labels: (batch_size, num_classes)  # multi-hot, but we treat each class separately
        """
        batch_size, num_classes = labels.shape
        loss = 0.0
        for c in range(num_classes):
            # For each class, get the binary label vector
            class_labels = labels[:, c]
            # Only consider samples with at least one positive and one negative
            if class_labels.sum() == 0 or class_labels.sum() == batch_size:
                continue  # skip if all are same
            # Compute similarity matrix
            sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
            # Mask out self-similarity
            mask = torch.eye(batch_size, device=embeddings.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            # For each anchor, positives are those with the same class label
            positives = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)).float()
            positives = positives * (~mask)  # remove self
            # For each anchor, NT-Xent loss
            exp_sim = torch.exp(sim_matrix)
            exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
            # For each anchor, sum over positives
            pos_exp_sim = (exp_sim * positives).sum(dim=1)
            # Avoid division by zero
            pos_exp_sim = torch.clamp(pos_exp_sim, min=1e-8)
            loss_c = -torch.log(pos_exp_sim / (exp_sim_sum.squeeze(1) + 1e-8))
            # Only consider anchors with at least one positive
            valid = (positives.sum(dim=1) > 0)
            loss += loss_c[valid].mean()
        return loss / num_classes

def plot_multilabel_embeddings(features, labels, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Convert multi-hot to class indices
    class_labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.get_cmap('viridis', 5)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=class_labels, cmap=cmap, alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=range(5))
    cbar.set_ticklabels(['Non-vuln', 'Reentrancy', 'Overflow', 'Timestamp', 'Unchecked Call'])
    plt.title(f'Embeddings Visualization - Epoch {epoch}')
    plt.savefig(os.path.join(save_dir, f'embeddings_epoch_{epoch}.png'))
    plt.close()
    
    return features_2d

def train_epoch(attention_model, contrastive_model, dataloader, memory_bank, criterion, attention_optimizer, contrastive_optimizer):
    attention_model.train()
    contrastive_model.train()
    total_loss = 0
    
    for ast_emb, byte_emb, cfg_emb, labels in dataloader:
        ast_emb = ast_emb.to(device)
        byte_emb = byte_emb.to(device)
        cfg_emb = cfg_emb.to(device)
        labels = labels.to(device)
        
        # First pass without attention
        embeddings = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=False)
        features, projections = contrastive_model(embeddings)
        ptr = memory_bank.update(projections.detach())
        loss1 = criterion(projections, labels)

        # Second pass with attention
        attended_emb = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=True)
        features, projections = contrastive_model(attended_emb)
        memory_bank.update(projections.detach(), ptr)
        loss2 = criterion(projections, labels)

        # Contrastive model update
        contrastive_optimizer.zero_grad()
        (loss1 + loss2).backward(retain_graph=True)
        contrastive_optimizer.step()
        
        # Third pass without attention for attention model update
        embeddings = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=False)
        features, projections = contrastive_model(embeddings)
        memory_bank.update(projections.detach(), ptr)
        loss1 = criterion(projections, labels)

        # Fourth pass with attention for attention model update
        attended_emb = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=True)
        features, projections = contrastive_model(attended_emb)
        memory_bank.update(projections.detach(), ptr)
        loss2 = criterion(projections, labels)

        # Attention model update
        attention_optimizer.zero_grad()
        (loss2 - loss1).backward()
        attention_optimizer.step()
        
        total_loss += (loss1 + loss2).item()
    
    return total_loss / len(dataloader)

def validate(attention_model, contrastive_model, dataloader, memory_bank, criterion):
    attention_model.eval()
    contrastive_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for ast_emb, byte_emb, cfg_emb, labels in dataloader:
            ast_emb = ast_emb.to(device)
            byte_emb = byte_emb.to(device)
            cfg_emb = cfg_emb.to(device)
            labels = labels.to(device)
            
            embeddings = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=True)
            _, projections = contrastive_model(embeddings)
            loss = criterion(projections, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def objective(trial, ast_emb, byte_emb, cfg_emb, labels):
    params = {
        'embed_dim': trial.suggest_int('embed_dim', 64, 128),
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 256),
        'output_dim': trial.suggest_int('output_dim', 64, 128),
        'temperature': trial.suggest_float('temperature', 0.1, 1.0),
        'similarity_threshold': trial.suggest_float('similarity_threshold', 0.3, 0.7),
        'lr_attention': trial.suggest_float('lr_attention', 1e-5, 1e-2, log=True),
        'lr_contrastive': trial.suggest_float('lr_contrastive', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 320, 512])
    }
    
    # Split data with multi-label stratification
    X_ast_train, X_ast_val, X_byte_train, X_byte_val, X_cfg_train, X_cfg_val, y_train, y_val = train_test_split(
        ast_emb, byte_emb, cfg_emb, labels, 
        test_size=0.2, stratify=labels.argmax(axis=1), random_state=42
    )
    
    train_dataset = MultiLabelVulnerabilityDataset(X_ast_train, X_byte_train, X_cfg_train, y_train)
    val_dataset = MultiLabelVulnerabilityDataset(X_ast_val, X_byte_val, X_cfg_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
    attention_model = MultiModalFusion(
        d1=ast_emb.shape[1],
        d2=byte_emb.shape[1],
        d3=cfg_emb.shape[1],
        d=params['embed_dim']
    ).to(device)
    
    contrastive_model = EnhancedContrastiveModel(
        params['embed_dim'], 
        params['hidden_dim'], 
        params['output_dim']
    ).to(device)
    
    criterion = SimpleNTXentLoss(temperature=params['temperature'])
    
    attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=params['lr_attention'])
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=params['lr_contrastive'])
    memory_bank = MemoryBank(len(train_dataset), params['output_dim'])
    
    num_epochs = 25
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(attention_model, contrastive_model, train_loader, 
                               memory_bank, criterion, attention_optimizer, contrastive_optimizer)
        val_loss = validate(attention_model, contrastive_model, val_loader, memory_bank, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def load_embeddings_multiclass(ast_path, byte_path, cfg_path, labels_path):
    # Load npz files
    ast_data = np.load(ast_path)
    byte_data = np.load(byte_path)
    cfg_data = np.load(cfg_path)
    labels_df = pd.read_csv(labels_path)

    # Clean filenames (remove suffixes)
    ast_filenames = [f.replace('_ast_processed', '').replace('.pt', '') for f in ast_data['filenames']]
    byte_filenames = [f.replace('_assembly', '').replace('.txt', '') for f in byte_data['filenames']]
    cfg_filenames = [f.replace('_cfg_processed', '').replace('.pt', '') for f in cfg_data['filenames']]
    labels_filenames = [str(f).split('.sol')[0] for f in labels_df['filename']]

    # Find intersection
    final_filenames = sorted(set(ast_filenames) & set(byte_filenames) & set(cfg_filenames) & set(labels_filenames))

    # Build index maps for fast lookup
    ast_idx = {f: i for i, f in enumerate(ast_filenames)}
    byte_idx = {f: i for i, f in enumerate(byte_filenames)}
    cfg_idx = {f: i for i, f in enumerate(cfg_filenames)}
    labels_idx = {f: i for i, f in enumerate(labels_filenames)}

    # Extract aligned data
    ast_emb = np.array([ast_data['embeddings'][ast_idx[f]] for f in final_filenames])
    byte_emb = np.array([byte_data['embeddings'][byte_idx[f]] for f in final_filenames])
    cfg_emb = np.array([cfg_data['embeddings'][cfg_idx[f]] for f in final_filenames])
    label_cols = ['integer_flows', 'reentrancy', 'timestamp', 'unchecked_low_level_calls', 'non_vulnerable']
    labels = np.array([labels_df.loc[labels_idx[f], label_cols].astype(int).to_numpy() for f in final_filenames])

    return ast_emb, byte_emb, cfg_emb, labels, final_filenames

def main():
    ast_path = 'embeddings_ast/ast_embeddings.npz'
    byte_path = 'embeddings_bytecode/bytecode_embeddings.npz'
    cfg_path = 'embeddings_cfg/cfg_embeddings.npz'
    labels_path = 'dataset/multivul_multiclass_test/Labels.csv'
    output_dir = Path('attended_embeddings')
    
    print("Loading embeddings...")
    ast_emb, byte_emb, cfg_emb, labels, filenames = load_embeddings_multiclass(ast_path, byte_path, cfg_path, labels_path)
    # labels should be shape (N, 5) for multi-label
    print("labels shape:", labels.shape)
    
    # Convert labels to multi-hot format (implement your conversion here)
    # labels = your_conversion_function(labels)  
    
    # Hyperparameter optimization with Optuna
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, ast_emb, byte_emb, cfg_emb, labels), 
                  n_trials=16,
                  show_progress_bar=True)
    
    best_params = study.best_params
    print(f"Best validation loss: {study.best_value:.4f}")
    
    # Full training with best parameters
    X_ast_train, X_ast_temp, X_byte_train, X_byte_temp, X_cfg_train, X_cfg_temp, y_train, y_temp = \
        train_test_split(ast_emb, byte_emb, cfg_emb, labels, 
                        test_size=0.2, stratify=labels.argmax(axis=1), random_state=42)
    
    X_ast_val, X_ast_test, X_byte_val, X_byte_test, X_cfg_val, X_cfg_test, y_val, y_test = \
        train_test_split(X_ast_temp, X_byte_temp, X_cfg_temp, y_temp, 
                        test_size=0.5, stratify=y_temp.argmax(axis=1), random_state=42)
    
    train_dataset = MultiLabelVulnerabilityDataset(X_ast_train, X_byte_train, X_cfg_train, y_train)
    val_dataset = MultiLabelVulnerabilityDataset(X_ast_val, X_byte_val, X_cfg_val, y_val)
    test_dataset = MultiLabelVulnerabilityDataset(X_ast_test, X_byte_test, X_cfg_test, y_test)
    complete_dataset = MultiLabelVulnerabilityDataset(ast_emb, byte_emb, cfg_emb, labels)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    complete_loader = DataLoader(complete_dataset, batch_size=best_params['batch_size'])
    
    attention_model = MultiModalFusion(
        d1=ast_emb.shape[1],
        d2=byte_emb.shape[1],
        d3=cfg_emb.shape[1],
        d=best_params['embed_dim']
    ).to(device)
    
    contrastive_model = EnhancedContrastiveModel(
        best_params['embed_dim'], 
        best_params['hidden_dim'], 
        best_params['output_dim']
    ).to(device)
    
    criterion = SimpleNTXentLoss(temperature=best_params['temperature'])
    
    attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=best_params['lr_attention'])
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=best_params['lr_contrastive'])
    memory_bank = MemoryBank(len(train_dataset), best_params['output_dim'])
    
    num_epochs = 60
    best_val_loss = float('inf')
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        train_loss = train_epoch(attention_model, contrastive_model, train_loader, 
                               memory_bank, criterion, attention_optimizer, contrastive_optimizer)
        val_loss = validate(attention_model, contrastive_model, val_loader, memory_bank, criterion)
        
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(attention_model.state_dict(), 'attended_embeddings/best_model.pt')
            torch.save(contrastive_model.state_dict(), 'ContrastiveLearning/best_model.pt')
            pbar.write(f'Epoch {epoch}: New best validation loss: {val_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            attention_model.eval()
            contrastive_model.eval()
            with torch.no_grad():
                all_features, all_labels = [], []
                for batch in complete_loader:
                    ast, byte, cfg, lbl = [x.to(device) for x in batch]
                    emb = attention_model(ast, byte, cfg, apply_attention=True)
                    feats, _ = contrastive_model(emb)
                    all_features.append(feats)
                    all_labels.append(lbl)
                
                features = torch.cat(all_features, dim=0)
                labels = torch.cat(all_labels, dim=0)
                plot_multilabel_embeddings(features, labels, epoch, 'ContrastiveLearning')
    
    # Final evaluation and embedding generation
    contrastive_model.load_state_dict(torch.load('ContrastiveLearning/best_model.pt'))
    test_loss = validate(attention_model, contrastive_model, test_loader, memory_bank, criterion)
    print(f'Final test loss: {test_loss:.4f}')
    
    # Generate and save embeddings
    print("Generating attended embeddings...")
    ast_tensor = torch.Tensor(ast_emb).to(device)
    byte_tensor = torch.Tensor(byte_emb).to(device)
    cfg_tensor = torch.Tensor(cfg_emb).to(device)
    
    attention_model.eval()
    with torch.no_grad():
        attended_emb = attention_model(ast_tensor, byte_tensor, cfg_tensor)
        _, contrasted_emb = contrastive_model(attended_emb)
        attended_emb = attended_emb.cpu().numpy() if hasattr(attended_emb, 'cpu') else np.array(attended_emb)
        contrasted_emb = contrasted_emb.cpu().numpy() if hasattr(contrasted_emb, 'cpu') else np.array(contrasted_emb)
    
    output_dir.mkdir(exist_ok=True)
    # Ensure all are numpy arrays on CPU
    if hasattr(attended_emb, 'cpu'):
        attended_emb = attended_emb.cpu().numpy()
    if hasattr(contrasted_emb, 'cpu'):
        contrasted_emb = contrasted_emb.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    elif isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if hasattr(filenames, 'cpu'):
        filenames = filenames.cpu().numpy()
    elif isinstance(filenames, list):
        filenames = np.array(filenames)
    np.savez(output_dir / 'attended_embeddings.npz',
             embeddings=attended_emb, labels=labels, filenames=filenames)
    np.savez(output_dir / 'contrasted_embeddings.npz',
             embeddings=contrasted_emb, labels=labels, filenames=filenames)
    
    print("Embeddings saved successfully")

if __name__ == '__main__':
    main()