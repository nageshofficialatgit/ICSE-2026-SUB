import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    make_scorer, precision_score, recall_score
)
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import cross_validate
import sys

class Tee(object):
    """A file-like object that writes to multiple files."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Flush to see output in real time

    def flush(self):
        for f in self.files:
            f.flush()

# Create a directory for outputs if it doesn't exist
output_dir = Path('classifier_outputs')
output_dir.mkdir(exist_ok=True)

# Open file for logging
log_file = open(output_dir / 'output.txt', 'w', encoding='utf-8')

# Save original stdout
original_stdout = sys.stdout

# Redirect stdout to both console and file
sys.stdout = Tee(sys.stdout, log_file)

def clean_filenames(filenames, suffix: str) -> list:
    """Clean filenames by removing suffixes and extensions"""
    if isinstance(filenames, pd.Series):
        # For CSV filenames, remove .sol extension
        filenames = [name.replace('.sol', '') for name in filenames]
    else:
        # For processed files, remove the specified suffix
        filenames = [name.replace(suffix, '') for name in filenames]
    return filenames

def find_matching_files(csv_base_names: list, processed_files: list) -> tuple[list, list, dict]:
    """Find which CSV files have corresponding processed files"""
    matching_files = []
    matching_labels = []
    label_to_file = {}  # Map from base name to actual processed filename
    
    for csv_base_name in csv_base_names:
        # Find all processed files that start with this base name
        matches = [f for f in processed_files if f.startswith(csv_base_name)]
        if matches:
            # Take the first matching file
            matching_files.append(matches[0])
            matching_labels.append(csv_base_name)
            label_to_file[csv_base_name] = matches[0]
            print(f"Found match: {csv_base_name} -> {matches[0]}")
    
    return matching_files, matching_labels, label_to_file

# Load embeddings and labels
#embeddings = np.load('attended_embeddings/attended_embeddings.npz')
embeddings = np.load('attended_embeddings/contrasted_embeddings.npz')
#embeddings = np.load('embeddings_ast/ast_embeddings.npz')
#embeddings = np.load('embeddings_bytecode/bytecode_embeddings.npz')
# embeddings = np.load('embeddings_cfg/cfg_embeddings.npz')
labels_df = pd.read_csv('dataset/pure_vul_test/Labels.csv')

# Get base names from CSV and processed files
csv_base_names = clean_filenames(labels_df['File'], '')
processed_filenames = clean_filenames(embeddings['filenames'], '_assembly')

print(f"Number of files in CSV: {len(csv_base_names)}")
print(f"First few CSV filenames: {csv_base_names[:5]}")
print(f"Number of processed files: {len(processed_filenames)}")

# Find matching files
matching_files, matching_labels, label_to_file = find_matching_files(csv_base_names, processed_filenames)
print(f"\nNumber of matching files found: {len(matching_labels)}")

# Create mapping from processed filename to index
processed_idx = {name: i for i, name in enumerate(processed_filenames)}
labels_idx = {name: i for i, name in enumerate(csv_base_names)}

# Extract aligned data
X = np.array([embeddings['embeddings'][processed_idx[label_to_file[name]]] for name in matching_labels])
y = np.array([labels_df.loc[labels_idx[name], 'Label'] for name in matching_labels])

# Balance the dataset
print("\nBefore balancing:")
print(f"Total samples: {len(y)}")
print(f"Vulnerable (1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.2f}%)")
print(f"Non-vulnerable (0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.2f}%)")

# Get indices of vulnerable and non-vulnerable samples
vul_indices = np.where(y == 1)[0]
non_vul_indices = np.where(y == 0)[0]

# Randomly select non-vulnerable samples to match vulnerable count
np.random.seed(42)
selected_non_vul = np.random.choice(non_vul_indices, size=len(vul_indices), replace=False)

# Combine indices and sort to maintain original order
balanced_indices = np.sort(np.concatenate([vul_indices, selected_non_vul]))

# Create balanced dataset
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]

print("\nAfter balancing:")
print(f"Total samples: {len(y_balanced)}")
print(f"Vulnerable (1): {np.sum(y_balanced == 1)} ({np.mean(y_balanced == 1)*100:.2f}%)")
print(f"Non-vulnerable (0): {np.sum(y_balanced == 0)} ({np.mean(y_balanced == 0)*100:.2f}%)")

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
)

# Print split configuration
print("\nDataset Split Configuration (70-30):")
print("=" * 50)
print("Training set (70%):")
print(f"Total samples: {len(y_train)}")
print(f"Vulnerable (1): {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.2f}%)")
print(f"Non-vulnerable (0): {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.2f}%)")
print(f"Class balance ratio (1:0): {np.sum(y_train == 1)}:{np.sum(y_train == 0)}")

print("\nTest set (30%):")
print(f"Total samples: {len(y_test)}")
print(f"Vulnerable (1): {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.2f}%)")
print(f"Non-vulnerable (0): {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.2f}%)")
print(f"Class balance ratio (1:0): {np.sum(y_test == 1)}:{np.sum(y_test == 0)}")

print("\nOverall dataset composition:")
print(f"Total samples: {len(y_balanced)}")
print(f"Training set percentage: {(len(y_train)/len(y_balanced))*100:.2f}%")
print(f"Test set percentage: {(len(y_test)/len(y_balanced))*100:.2f}%")

# SVM Classifier
print("\nTraining SVM on 70-30 split...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, degree=64)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
print(f"SVM Results:\nAccuracy: {svm_acc:.4f}\nF1 Score: {svm_f1:.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

# Random Forest Classifier
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"Random Forest Results:\nAccuracy: {rf_acc:.4f}\nF1 Score: {rf_f1:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# XGBoost Classifier
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
print(f"XGBoost Results:\nAccuracy: {xgb_acc:.4f}\nF1 Score: {xgb_f1:.4f}")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# MLP Classifier (PyTorch)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.layers(x))

print("\nTraining MLP...")
mlp = MLP(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

# Convert to tensors
train_dataset = TensorDataset(
    torch.FloatTensor(X_train), 
    torch.FloatTensor(y_train).unsqueeze(1)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = mlp(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    test_tensor = torch.FloatTensor(X_test)
    mlp_pred = (mlp(test_tensor) > 0.5).float().squeeze().numpy()
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred)
    
print(f"MLP Results:\nAccuracy: {mlp_acc:.4f}\nF1 Score: {mlp_f1:.4f}")

# Visualization
def plot_embeddings(X, y, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Handle small sample sizes for t-SNE
    n_samples = len(X)
    perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
    
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)
        
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette='viridis')
        plt.title(f'{title} (perplexity={perplexity})')
    except Exception as e:
        print(f"Could not create t-SNE plot: {str(e)}")
        return
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Save plot instead of showing
    output_dir = Path('classifier_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{filename}.png')
    plt.close()

# Plot visualizations for all models
plot_embeddings(X_test, y_test, 'True Labels Visualization', 'true_labels')
plot_embeddings(X_test, svm_pred, 'SVM Predictions Visualization', 'svm_predictions')
plot_embeddings(X_test, rf_pred, 'Random Forest Predictions Visualization', 'rf_predictions')
plot_embeddings(X_test, xgb_pred, 'XGBoost Predictions Visualization', 'xgb_predictions')
plot_embeddings(X_test, mlp_pred, 'MLP Predictions Visualization', 'mlp_predictions')

# Confusion matrices for all models with detailed annotations
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# Function to plot confusion matrix with detailed annotations
def plot_confusion_matrix(y_true, y_pred, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    # Add percentage annotations
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i,j] / total * 100
            ax.text(j+0.5, i+0.7, f'\n({percentage:.1f}%)', 
                   ha='center', va='center', color='black')

plot_confusion_matrix(y_test, svm_pred, ax[0,0], 'SVM Confusion Matrix')
plot_confusion_matrix(y_test, rf_pred, ax[0,1], 'Random Forest Confusion Matrix')
plot_confusion_matrix(y_test, xgb_pred, ax[1,0], 'XGBoost Confusion Matrix')
plot_confusion_matrix(y_test, mlp_pred, ax[1,1], 'MLP Confusion Matrix')

plt.tight_layout()
plt.savefig('classifier_outputs/confusion_matrices.png')
plt.close()

# Feature importance plots for tree-based models
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(20)
sns.barplot(data=rf_importance, x='importance', y='feature', ax=ax[0])
ax[0].set_title('Random Forest Feature Importance (Top 20)')

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)
sns.barplot(data=xgb_importance, x='importance', y='feature', ax=ax[1])
ax[1].set_title('XGBoost Feature Importance (Top 20)')

plt.tight_layout()
plt.savefig('classifier_outputs/feature_importance.png')
plt.close()

# Save models
torch.save(mlp.state_dict(), 'classifier_outputs/mlp_classifier.pth')
import joblib
joblib.dump(svm, 'classifier_outputs/svm_classifier.pkl')
joblib.dump(rf, 'classifier_outputs/rf_classifier.pkl')
joblib.dump(xgb_model, 'classifier_outputs/xgb_classifier.pkl')

# Print summary of all models
print("\nModel Performance Summary:")
print(f"{'Model':<15} {'Accuracy':<10} {'F1 Score':<10}")
print("-" * 35)
print(f"{'SVM':<15} {svm_acc:.4f}     {svm_f1:.4f}")
print(f"{'Random Forest':<15} {rf_acc:.4f}     {rf_f1:.4f}")
print(f"{'XGBoost':<15} {xgb_acc:.4f}     {xgb_f1:.4f}")
print(f"{'MLP':<15} {mlp_acc:.4f}     {mlp_f1:.4f}")

# Function to perform k-fold cross validation
def perform_cross_validation(X, y, models, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {}
    
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    
    for name, model in models.items():
        print(f"\nPerforming {n_splits}-fold cross validation for {name}...")
        print(f"Total samples: {len(X)}, Samples per fold: ~{len(X)//n_splits}")
        
        # For PyTorch models, we'll use a custom cross-validation approach
        if name == 'MLP':
            fold_scores = {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                print(f"\nFold {fold}/{n_splits}")
                print(f"Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
                
                # Train MLP
                mlp = MLP(X.shape[1])
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
                
                # Convert to tensors
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_fold), 
                    torch.FloatTensor(y_train_fold).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # Training loop
                for epoch in range(100):
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = mlp(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(X_val_fold)
                    y_pred = (mlp(val_tensor) > 0.5).float().squeeze().numpy()
                
                # Calculate metrics
                fold_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                fold_scores['f1'].append(f1_score(y_val_fold, y_pred))
                fold_scores['precision'].append(precision_score(y_val_fold, y_pred))
                fold_scores['recall'].append(recall_score(y_val_fold, y_pred))
                
                print(f"Fold {fold} Results:")
                print(f"  Accuracy: {fold_scores['accuracy'][-1]:.4f}")
                print(f"  F1 Score: {fold_scores['f1'][-1]:.4f}")
                print(f"  Precision: {fold_scores['precision'][-1]:.4f}")
                print(f"  Recall: {fold_scores['recall'][-1]:.4f}")
            
            # Calculate mean and std for each metric
            cv_results[name] = {
                metric: {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
                for metric, scores in fold_scores.items()
            }
        else:
            # For sklearn models, use cross_validate
            cv_scores = cross_validate(
                model, X, y,
                cv=kf,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1  # Use all available cores
            )
            
            cv_results[name] = {
                metric: {
                    'mean': np.mean(cv_scores[f'test_{metric}']),
                    'std': np.std(cv_scores[f'test_{metric}']),
                    'train_mean': np.mean(cv_scores[f'train_{metric}']),
                    'train_std': np.std(cv_scores[f'train_{metric}'])
                }
                for metric in scoring.keys()
            }
            
            # Print detailed results for each fold
            print(f"\nDetailed {n_splits}-fold results for {name}:")
            for metric in scoring.keys():
                print(f"\n{metric.capitalize()}:")
                print("Fold\tTrain\t\tTest")
                print("-" * 30)
                for fold in range(n_splits):
                    train_score = cv_scores[f'train_{metric}'][fold]
                    test_score = cv_scores[f'test_{metric}'][fold]
                    print(f"{fold+1}\t{train_score:.4f}\t\t{test_score:.4f}")
    
    return cv_results

# Define models for cross validation
models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, degree=64),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'MLP': None  # Will be handled separately in cross validation
}

# Perform cross validation with 10 folds
print("\nStarting 10-fold cross validation...")
cv_results = perform_cross_validation(X_balanced, y_balanced, models, n_splits=10)

# Print cross validation results with more detailed formatting
print("\nCross Validation Results Summary (10-fold):")
print("=" * 80)
for model_name, results in cv_results.items():
    print(f"\n{model_name}:")
    print("-" * 40)
    for metric, scores in results.items():
        if 'train_mean' in scores:
            print(f"{metric.capitalize()}:")
            print(f"  Test  - Mean: {scores['mean']:.4f} ± {scores['std']:.4f}")
            print(f"  Train - Mean: {scores['train_mean']:.4f} ± {scores['train_std']:.4f}")
            print(f"  Train-Test Gap: {abs(scores['train_mean'] - scores['mean']):.4f}")
        else:
            print(f"{metric.capitalize()}: Mean: {scores['mean']:.4f} ± {scores['std']:.4f}")

# Plot cross validation results with updated title
def plot_cv_results(cv_results, metric='accuracy'):
    plt.figure(figsize=(12, 6))
    
    models = list(cv_results.keys())
    means = [results[metric]['mean'] for results in cv_results.values()]
    stds = [results[metric]['std'] for results in cv_results.values()]
    
    # For models with train scores, plot both
    train_means = []
    train_stds = []
    for results in cv_results.values():
        if 'train_mean' in results[metric]:
            train_means.append(results[metric]['train_mean'])
            train_stds.append(results[metric]['train_std'])
    
    x = np.arange(len(models))
    width = 0.35
    
    if train_means:
        plt.bar(x - width/2, train_means, width, label='Train', yerr=train_stds, capsize=5)
        plt.bar(x + width/2, means, width, label='Test', yerr=stds, capsize=5)
    else:
        plt.bar(x, means, yerr=stds, capsize=5)
    
    plt.xlabel('Models')
    plt.ylabel(f'{metric.capitalize()} Score')
    plt.title(f'{metric.capitalize()} Scores from 10-Fold Cross Validation')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot with updated filename
    output_dir = Path('classifier_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'cv_10fold_{metric}_scores.png')
    plt.close()

# Plot results for each metric with updated filenames
for metric in ['accuracy', 'f1', 'precision', 'recall']:
    plot_cv_results(cv_results, metric)

# Restore stdout and close the log file
sys.stdout = original_stdout
log_file.close()

print(f"Terminal output was also saved to {output_dir / 'output.txt'}")
# Continue with the original train-test split for final model training and evaluation 