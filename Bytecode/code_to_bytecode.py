import os
import subprocess

def compile_sol_to_bytecode(sol_folder_path, output_folder_path):
    """
    Converts Solidity (.sol) files in the given folder to their corresponding bytecode
    and saves each bytecode in a separate text file in the output folder.

    Args:
        sol_folder_path (str): Path to the folder containing .sol files.
        output_folder_path (str): Path to the folder to save the bytecode files.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for root, _, files in os.walk(sol_folder_path):
        for file in files:
            if file.endswith(".sol"):
                sol_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder_path, file.replace(".sol", ".txt"))
                try:
                    # Compile Solidity file and extract bytecode
                    result = subprocess.run(
                        ["solc", "--bin", sol_file_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    if result.returncode == 0:
                        bytecode = extract_bytecode(result.stdout)
                        if bytecode:
                            # Save the bytecode to a file
                            with open(output_file_path, "w") as f:
                                f.write(bytecode)
                            print(f"Bytecode saved for {file}")
                        else:
                            print(f"No bytecode found for {file}")
                    else:
                        print(f"Error compiling {file}: {result.stderr}")
                except FileNotFoundError:
                    raise RuntimeError("solc compiler not found. Ensure it is installed and in your PATH.")

def extract_bytecode(solc_output):
    """
    Extracts bytecode from the output of solc.
    """
    lines = solc_output.splitlines()
    for i, line in enumerate(lines):
        if "Binary:" in line:
            return lines[i + 1].strip()  # Bytecode appears on the next line
    return None

if __name__ == "__main__":
    # Example usage
    sol_folder_path = "dataset"  # Replace with the actual folder path
    output_folder_path = "Bytecode/Dataset_Bytecode"  # Replace with the desired output folder path

    compile_sol_to_bytecode(sol_folder_path, output_folder_path)
