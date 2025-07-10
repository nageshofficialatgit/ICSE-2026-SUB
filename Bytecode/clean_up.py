import os
import pandas as pd

# Paths (update if needed)
directory = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\Bytecode\Dataset_Assembly"
csv_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\multivul_multiclass_test\Labels.csv"

# Read the CSV and get the set of allowed base filenames (before .sol)
df = pd.read_csv(csv_path)
allowed_files = set(str(f).split('.sol')[0] for f in df['filename'])

# List all files in the directory
for filename in os.listdir(directory):
    # Only the part before '_assembly.txt' is used for comparison
    if '_assembly.txt' in filename:
        file_id = filename.split('_assembly.txt')[0]
    else:
        file_id = os.path.splitext(filename)[0]
    # If file_id is not in allowed_files, remove the file
    if file_id not in allowed_files:
        file_path = os.path.join(directory, filename)
        print(f"Removing: {file_path}")
        os.remove(file_path)

print("Cleanup complete.")