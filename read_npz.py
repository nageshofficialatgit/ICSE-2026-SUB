import numpy as np

def npz_to_txt(npz_filepath, output_filepath):
    """
    Convert NPZ file to TXT file
    
    Args:
        npz_filepath (str): Path to input NPZ file
        output_filepath (str): Path to output TXT file
    """
    # Load the NPZ file
    data = np.load(npz_filepath, allow_pickle=True)
    
    # Open the output text file
    with open(output_filepath, 'w') as f:
        # Iterate through all arrays in the NPZ file
        for key in data.files:
            # Write the array name as a header
            f.write(f"=== Array: {key} ===\n")
            
            # Get the array
            array = data[key]
            
            print(array.shape)
            
            # If the array is 1D, write it as a single line
            if array.ndim == 1:
                np.savetxt(f, array.reshape(1, -1), fmt='%s', delimiter='\t')
            else:
                # For multi-dimensional arrays, write them with proper formatting
                np.savetxt(f, array, fmt='%s', delimiter='\t')
            
            f.write('\n')  # Add blank line between arrays

if __name__ == "__main__":
    # Example usage
    # input_file = "attended_embeddings/attended_embeddings.npz"
    # output_file = "attended_embeddings/attended_embeddings.txt"
    # input_file = "attended_embeddings/contrasted_embeddings.npz"
    # output_file = "attended_embeddings/contrasted_embeddings.txt"
    # input_file = "concatenated_embeddings/concatenated_embeddings.npz"
    # output_file = "concatenated_embeddings/concatenated_embeddings.txt"
    input_file = "embeddings_ast/ast_embeddings.npz"
    output_file = "embeddings_ast/ast_embeddings.txt"
    # input_file = "embeddings_bytecode/bytecode_embeddings.npz"
    # output_file = "embeddings_bytecode/bytecode_embeddings.txt"
    # input_file = "embeddings_cfg/cfg_embeddings.npz"
    # output_file = "embeddings_cfg/cfg_embeddings.txt"
    
    try:
        npz_to_txt(input_file, output_file)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
