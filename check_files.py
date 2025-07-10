import os
import pandas as pd

def remove_extension(filename):
    """Remove .sol extension if present"""
    if filename.lower().endswith('.sol'):
        return filename[:-4]
    return filename

def check_files(folder_path, labels_path):
    try:
        # Read the labels CSV
        labels_df = pd.read_csv(labels_path)
        
        # Get list of files from CSV (without .sol)
        csv_files = set(remove_extension(f) for f in labels_df['File'].tolist())
        
        # Get list of files from folder (without .sol)
        folder_files = set(remove_extension(f) for f in os.listdir(folder_path))
        
        # Find missing files in folder
        missing_in_folder = csv_files - folder_files
        
        # Find extra files in folder not in CSV
        missing_in_csv = folder_files - csv_files
        
        # Print results
        if missing_in_folder:
            print("Files in CSV but missing in folder:")
            for file in sorted(missing_in_folder):
                print(f" - {file}.sol")
        else:
            print("All files in CSV are present in the folder.")
        
        if missing_in_csv:
            print("\nFiles in folder but missing in CSV:")
            for file in sorted(missing_in_csv):
                print(f" - {file}.sol")
        else:
            print("\nAll files in folder are present in the CSV.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    folder_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\pure_vul_test"
    labels_path = os.path.join(folder_path, "labels.csv")
    
    check_files(folder_path, labels_path) 