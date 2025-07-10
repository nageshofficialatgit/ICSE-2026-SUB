import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from pathlib import Path
import plotly.express as px

# --- CONFIG ---
csv_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\multivul_multiclass_test\Labels.csv"
label_cols = ['integer_flows', 'reentrancy', 'timestamp', 'unchecked_low_level_calls', 'non_vulnerable']

embedding_files = {
    "contrasted": r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\attended_embeddings\contrasted_embeddings.npz",
    "attended": r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\attended_embeddings\attended_embeddings.npz"
}

# --- LOAD LABELS ---
df = pd.read_csv(csv_path)
df['file_id'] = df['filename'].apply(lambda x: x.split('.sol')[0])
label_dict = {row['file_id']: tuple(row[label_cols].astype(int)) for _, row in df.iterrows()}

def get_labels_and_embeddings(embeddings_path):
    data = np.load(embeddings_path)
    X = data['embeddings']
    filenames = [os.path.basename(f).replace('_ast_processed', '').replace('_assembly', '').replace('_cfg_processed', '').replace('.pt', '').replace('.txt', '') for f in data['filenames']]
    # Align labels
    labels = []
    used_filenames = []
    for fname in filenames:
        if fname in label_dict:
            labels.append(label_dict[fname])
            used_filenames.append(fname)
        else:
            print(f"Warning: {fname} not found in CSV, skipping.")
    labels = np.array(labels)
    X = X[:len(labels)]
    y = np.array([np.argmax(l) for l in labels])
    return X, y, used_filenames

def plot_tsne(X, y, filenames, title, filename):
    n_samples = len(X)
    perplexity = min(30, n_samples - 1)
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)
        df_plot = pd.DataFrame({
            'tSNE-1': X_2d[:, 0],
            'tSNE-2': X_2d[:, 1],
            'Label': y.astype(str),
            'Filename': filenames
        })
        fig = px.scatter(
            df_plot, x='tSNE-1', y='tSNE-2', color='Label',
            hover_data=['Filename'],
            title=f'{title} (perplexity={perplexity})',
            labels={'Label': 'Class'}
        )
        output_dir = Path('tsne_outputs')
        output_dir.mkdir(exist_ok=True)
        fig.write_html(str(output_dir / f'{filename}.html'))
    except Exception as e:
        print(f"Could not create t-SNE plot: {str(e)}")
        return

for name, path in embedding_files.items():
    print(f"Processing {name} embeddings...")
    X, y, used_filenames = get_labels_and_embeddings(path)
    plot_tsne(X, y, used_filenames, f't-SNE of {name.capitalize()} Embeddings', f'tsne_{name}_embeddings')

print("t-SNE plots saved in tsne_outputs/")