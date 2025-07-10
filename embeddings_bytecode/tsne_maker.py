import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Paths
embeddings_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\embeddings_bytecode\bytecode_embeddings.npz"
csv_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\multivul_multiclass_test\Labels.csv"

# Load embeddings
data = np.load(embeddings_path)
embeddings = data['embeddings']
filenames = [f.replace('_assembly', '').replace('.txt', '') for f in data['filenames']]

# Load CSV
df = pd.read_csv(csv_path)
df['file_id'] = df['filename'].apply(lambda x: x.split('.sol')[0])

# Build a mapping from file_id to label tuple
label_cols = ['integer_flows', 'reentrancy', 'timestamp', 'unchecked_low_level_calls', 'non_vulnerable']
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
embeddings = embeddings[:len(labels)]  # Ensure same length

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Assign unique color to each unique label combination
label_tuples = [tuple(l) for l in labels]
unique_combos = sorted(set(label_tuples))
color_list = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
color_map = {combo: color_list[i % len(color_list)] for i, combo in enumerate(unique_combos)}

print("t-SNE color mapping (label combination -> color):")
for combo, color in color_map.items():
    print(f"{combo}: {color}")

point_colors = [color_map[combo] for combo in label_tuples]

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, alpha=0.7)
handles = [mpatches.Patch(color=color_map[combo], label=str(combo)) for combo in unique_combos]
plt.legend(handles=handles, title="Label Combination", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of Bytecode Embeddings (colored by label combination)')
plt.tight_layout()
plt.savefig('tsne_bytecode_embeddings.png')
plt.show()