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

from combine_embeddings import MultiModalFusion, load_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VulnerabilityDataset(Dataset):
    def __init__(self, ast_emb, byte_emb, cfg_emb, labels):
        self.ast_emb = torch.FloatTensor(ast_emb)
        self.byte_emb = torch.FloatTensor(byte_emb)
        self.cfg_emb = torch.FloatTensor(cfg_emb)
        self.labels = torch.LongTensor(labels)
        
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
        
    def update(self, new_features, ptr = None):
        if ptr == None:
            ptr = self.ptr
        batch_size = new_features.size(0)
        self.features[ptr:ptr + batch_size] = new_features
        self.ptr = (ptr + batch_size) % self.size
        return ptr
        
    def get_all_features(self):
        return self.features

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor_features, memory_bank_features, labels):
        anchor_features = F.normalize(anchor_features, dim=1)
        # similarity_matrix = torch.matmul(anchor_features, memory_bank_features.T) / self.temperature
        similarity_matrix = torch.matmul(anchor_features, anchor_features.T) / self.temperature
        
        # Create mask for positive pairs (same vulnerability label)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # For each anchor, we want exp(pos_pair) / sum(exp(all_pairs))
        exp_sim = torch.exp(similarity_matrix)
        
        # Zero out diagonal elements
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        # Calculate positive pairs loss
        pos_sim = similarity_matrix.masked_fill(~positive_mask, 0)
        pos_exp = torch.exp(pos_sim)
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Calculate final loss
        loss = -torch.log(pos_exp / (denominator + 1e-8)).mean()
        
        return loss

def plot_embeddings(features, labels, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Convert to numpy for t-SNE
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='RdYlGn', alpha=0.7)
    plt.colorbar(scatter)
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
        
        # Calculating L1 i.e. Loss without Attention
        embeddings = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=False)
        features, projections = contrastive_model(embeddings)
        ptr = memory_bank.update(projections.detach())
        
        loss1 = criterion(projections, memory_bank.get_all_features().to(device), labels)

        # Calculating L2 i.e. Loss with Attention
        attended_emb = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=True)
        features, projections = contrastive_model(attended_emb)
        memory_bank.update(projections.detach(), ptr)

        loss2 = criterion(projections, memory_bank.get_all_features().to(device), labels)

        # Updating Contrastive Model
        contrastive_optimizer.zero_grad()
        (loss1 + loss2).backward(retain_graph=True)
        contrastive_optimizer.step()
        
        # Calculating L1 i.e. Loss without Attention
        embeddings = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=False)
        features, projections = contrastive_model(embeddings)
        memory_bank.update(projections.detach(), ptr)
        
        loss1 = criterion(projections, memory_bank.get_all_features().to(device), labels)

        # Calculating L2 i.e. Loss with Attention
        attended_emb = attention_model(ast_emb, byte_emb, cfg_emb, apply_attention=True)
        features, projections = contrastive_model(attended_emb)
        memory_bank.update(projections.detach(), ptr)

        loss2 = criterion(projections, memory_bank.get_all_features().to(device), labels)

        # Updating Attentions Model
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
            features, projections = contrastive_model(embeddings)

            loss = criterion(projections, memory_bank.get_all_features().to(device), labels)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def objective(trial, ast_emb, byte_emb, cfg_emb, labels):
    # Define hyperparameters to optimize
    params = {
        'embed_dim': trial.suggest_int('embed_dim', 64, 128),
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 256),
        'output_dim': trial.suggest_int('output_dim', 64, 128),
        'temperature': trial.suggest_float('temperature', 0.1, 1.0),
        'lr_attention': trial.suggest_float('lr_attention', 1e-5, 1e-2, log=True),
        'lr_contrastive': trial.suggest_float('lr_contrastive', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 320, 512])
    }
    
    # Split data
    X_ast_train, X_ast_val, X_byte_train, X_byte_val, X_cfg_train, X_cfg_val, y_train, y_val = train_test_split(ast_emb, byte_emb, cfg_emb, labels, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = VulnerabilityDataset(X_ast_train, X_byte_train, X_cfg_train, y_train)
    val_dataset = VulnerabilityDataset(X_ast_val, X_byte_val, X_cfg_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
    # Initialize model and training components
    attention_model = MultiModalFusion(
        d1=ast_emb.shape[1],
        d2=byte_emb.shape[1],
        d3=cfg_emb.shape[1],
        d=params['embed_dim']
    ).to(device)
    contrastive_model = ContrastiveModel(params['embed_dim'], params['hidden_dim'], params['output_dim']).to(device)
    criterion = NTXentLoss(temperature=params['temperature'])
    attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=params['lr_attention'])
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=params['lr_contrastive'])
    memory_bank = MemoryBank(len(train_dataset), params['output_dim'])
    
    # Quick training for optimization
    num_epochs = 25
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(attention_model, contrastive_model, train_loader, memory_bank, criterion, attention_optimizer, contrastive_optimizer)
        val_loss = validate(attention_model, contrastive_model, val_loader, memory_bank, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def main():
    # Paths
    ast_path = 'embeddings_ast/ast_embeddings.npz'
    byte_path = 'embeddings_bytecode/bytecode_embeddings.npz'
    cfg_path = 'embeddings_cfg/cfg_embeddings.npz'
    labels_path = 'dataset/pure_vul_test/Labels.csv'
    
    output_dir = Path('attended_embeddings')
    
    # Load embeddings
    print("Loading embeddings...")
    ast_emb, byte_emb, cfg_emb, labels, filenames = load_embeddings(ast_path, byte_path, cfg_path, labels_path)
    
    """
    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, ast_emb, byte_emb, cfg_emb, labels), 
                  n_trials=16,  # You can adjust this number
                  show_progress_bar=True)
    
    # Get best parameters from optimization
    best_params = {
        'embed_dim': study.best_params['embed_dim'],
        'hidden_dim': study.best_params['hidden_dim'],
        'output_dim': study.best_params['output_dim'],
        'temperature': study.best_params['temperature'],
        'lr_attention': study.best_params['lr_attention'],
        'lr_contrastive': study.best_params['lr_contrastive'],
        'batch_size': study.best_params['batch_size']
    }
    
    # Save optimization results
    optimization_results = {
        'best_params': best_params,
        'best_value': study.best_value,
        'optimization_history': [
            {
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]
    }
    
    # Save optimization results to JSON
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=4)
	
    print("Best hyperparameters found:", best_params)
    print(f"Best validation loss: {study.best_value:.4f}")
    """
    
    best_params = {
        'embed_dim': 96,
        'hidden_dim': 256,
        'output_dim': 128,
        'temperature': 0.5,
        'lr_attention': 5e-5,
        'lr_contrastive': 1e-4,
        'batch_size': 128
    }
    # """
    
    # Split data
    X_ast_train, X_ast_temp, X_byte_train, X_byte_temp, X_cfg_train, X_cfg_temp, y_train, y_temp = train_test_split(
        ast_emb, byte_emb, cfg_emb, labels, test_size=0.2, random_state=42
    )
    X_ast_val, X_ast_test, X_byte_val, X_byte_test, X_cfg_val, X_cfg_test, y_val, y_test = train_test_split(
        X_ast_temp, X_byte_temp, X_cfg_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create datasets
    train_dataset = VulnerabilityDataset(X_ast_train, X_byte_train, X_cfg_train, y_train)
    val_dataset = VulnerabilityDataset(X_ast_val, X_byte_val, X_cfg_val, y_val)
    test_dataset = VulnerabilityDataset(X_ast_test, X_byte_test, X_cfg_test, y_test)
    complete_dataset = VulnerabilityDataset(ast_emb, byte_emb, cfg_emb, labels)
    
    # Create data loaders with optimized batch size
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    complete_loader = DataLoader(complete_dataset, batch_size=best_params['batch_size'])
    
    # Initialize model and training components
    attention_model = MultiModalFusion(
        d1=ast_emb.shape[1],
        d2=byte_emb.shape[1],
        d3=cfg_emb.shape[1],
        d=best_params['embed_dim']
    ).to(device)
    
    contrastive_model = ContrastiveModel(
        best_params['embed_dim'], 
        best_params['hidden_dim'], 
        best_params['output_dim']
    ).to(device)
    
    criterion = NTXentLoss(temperature=best_params['temperature'])
    attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=best_params['lr_attention'])
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=best_params['lr_contrastive'])
    memory_bank = MemoryBank(len(train_dataset), best_params['output_dim'])
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    # Rest of your training loop remains the same
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        train_loss = train_epoch(attention_model, contrastive_model, train_loader, memory_bank, 
                               criterion, attention_optimizer, contrastive_optimizer)
        val_loss = validate(attention_model, contrastive_model, val_loader, memory_bank, criterion)
        
        # Update progress bar description
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })
        
        # Get embeddings for visualization
        attention_model.eval()
        contrastive_model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            for ast_emb_temp, byte_emb_temp, cfg_emb_temp, labels_temp in complete_loader:
                ast_emb_temp = ast_emb_temp.to(device)
                byte_emb_temp = byte_emb_temp.to(device)
                cfg_emb_temp = cfg_emb_temp.to(device)
                labels_temp = labels_temp.to(device)
                
                embeddings = attention_model(ast_emb_temp, byte_emb_temp, cfg_emb_temp, apply_attention=True)
                features, projections = contrastive_model(embeddings)
                all_features.append(features)
                all_labels.append(labels_temp)
            
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(attention_model.state_dict(), 'attended_embeddings/best_model.pt')
            torch.save(contrastive_model.state_dict(), 'ContrastiveLearning/best_model.pt')
            # Use tqdm.write to prevent breaking progress bar
            pbar.write(f'Epoch {epoch}: New best validation loss: {val_loss:.4f}')
        
        # Plot embeddings periodically
        if (epoch + 1) % 10 == 0:  # Plot every 10 epochs
            plot_embeddings(all_features, all_labels, epoch, 'ContrastiveLearning')
    
    # Final evaluation
    contrastive_model.load_state_dict(torch.load('ContrastiveLearning/best_model.pt'))
    test_loss = validate(attention_model, contrastive_model, test_loader, memory_bank, criterion)
    print(f'Final test loss: {test_loss:.4f}')
    
    # Generate attended embeddings
    print("Generating attended embeddings...")
    
    # Convert to tensors
    ast_tensor = torch.Tensor(ast_emb).to(device)
    byte_tensor = torch.Tensor(byte_emb).to(device)
    cfg_tensor = torch.Tensor(cfg_emb).to(device)
    
    attention_model.eval()
    with torch.no_grad():
        attended_emb = attention_model.forward(ast_tensor, byte_tensor, cfg_tensor)
        _, contrasted_emb = contrastive_model.forward(attended_emb)
        attended_emb = attended_emb.cpu().numpy()
        contrasted_emb = contrasted_emb.cpu().numpy()
    
    # Save attended embeddings
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'attended_embeddings.npz'
    np.savez(output_path,
             embeddings=attended_emb,
             labels=labels,
             filenames=filenames)
    
    print(f"Attended embeddings saved to {output_path}")
    print(f"Attended embedding shape: {attended_emb.shape}")
    
    # Save contrasted embeddings
    output_path = output_dir / 'contrasted_embeddings.npz'
    np.savez(output_path,
             embeddings=contrasted_emb,
             labels=labels,
             filenames=filenames)
    
    print(f"Contrasted embeddings saved to {output_path}")
    print(f"Contrasted embedding shape: {contrasted_emb.shape}")

if __name__ == '__main__':
    main()