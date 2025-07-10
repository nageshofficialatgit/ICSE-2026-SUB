import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import os
from pathlib import Path
import joblib

# --- CONFIG ---
embeddings_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\attended_embeddings\contrasted_embeddings.npz"
csv_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\multivul_multiclass_test\Labels.csv"
label_cols = ['integer_flows', 'reentrancy', 'timestamp', 'unchecked_low_level_calls', 'non_vulnerable']

# --- LOAD DATA ---
embeddings = np.load(embeddings_path)
X = embeddings['embeddings']
filenames = [os.path.basename(f).replace('_ast_processed', '').replace('_assembly', '').replace('_cfg_processed', '').replace('.pt', '').replace('.txt', '') for f in embeddings['filenames']]

df = pd.read_csv(csv_path)
df['file_id'] = df['filename'].apply(lambda x: x.split('.sol')[0])

# Build mapping from file_id to label vector
label_dict = {row['file_id']: tuple(row[label_cols].astype(int)) for _, row in df.iterrows()}

# Align labels to embeddings order
labels = []
used_filenames = []
for fname in filenames:
    if fname in label_dict:
        labels.append(label_dict[fname])
        used_filenames.append(fname)
    else:
        print(f"Warning: {fname} not found in CSV, skipping.")

labels = np.array(labels)
X = X[:len(labels)]  # Ensure same length

# --- MULTICLASS LABELS ---
# Convert multi-label to multiclass (class = index of 1 in the label vector)
y = np.array([np.argmax(l) for l in labels])

# --- DYNAMIC CLASS SELECTION ---
print("Available classes:")
for idx, name in enumerate(label_cols):
    print(f"{idx}: {name}")
selected = input("Enter class indices to consider (comma-separated, e.g. 1,2,4): ")
selected_classes = sorted([int(x.strip()) for x in selected.split(',') if x.strip().isdigit()])
print(f"Selected classes: {selected_classes} -> {[label_cols[i] for i in selected_classes]}")

mask = np.isin(y, selected_classes)
X = X[mask]
labels = labels[mask]
y = y[mask]
used_filenames = [fname for i, fname in enumerate(used_filenames) if mask[i]]

# Reindex classes to contiguous 0...N-1
class_mapping = {orig: new for new, orig in enumerate(selected_classes)}
y = np.array([class_mapping[cls] for cls in y])

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# --- SVM CLASSIFIER ---
print("Training SVM...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
print(f"SVM Results:\nAccuracy: {svm_acc:.4f}\nF1 Score: {svm_f1:.4f}")

# --- MLP CLASSIFIER (scikit-learn) ---
print("\nTraining MLP (scikit-learn)...")
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
print(f"MLP Results:\nAccuracy: {mlp_acc:.4f}\nF1 Score: {mlp_f1:.4f}")

# --- VISUALIZATION ---
def plot_embeddings(X, y, title, filename):
    plt.figure(figsize=(10, 6))
    n_samples = len(X)
    perplexity = min(30, n_samples - 1)
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette='tab10')
        plt.title(f'{title} (perplexity={perplexity})')
    except Exception as e:
        print(f"Could not create t-SNE plot: {str(e)}")
        return
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    output_dir = Path('classifier_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{filename}.png')
    plt.close()

plot_embeddings(X_test, y_test, 'True Labels Visualization', 'true_labels')
plot_embeddings(X_test, svm_pred, 'SVM Predictions Visualization', 'svm_predictions')
plot_embeddings(X_test, mlp_pred, 'MLP Predictions Visualization', 'mlp_predictions')

# --- CONFUSION MATRICES ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', ax=ax[0])
ax[0].set_title('SVM Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, mlp_pred), annot=True, fmt='d', ax=ax[1])
ax[1].set_title('MLP Confusion Matrix')
plt.savefig('classifier_outputs/confusion_matrices.png')
plt.close()

# --- SAVE MODELS ---
joblib.dump(svm, 'classifier_outputs/svm_classifier.pkl')
joblib.dump(mlp, 'classifier_outputs/mlp_classifier.pkl')