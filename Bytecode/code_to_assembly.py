import os
import re
import subprocess

version = '4'
command = ['solc-select', 'use', '0.4.26']
result = subprocess.run(command, capture_output=True, text=True, check=True)

def extract_solc_version(sol_file_path):
    """Extract the solc version from any pragma statement in the file"""
    try:
        with open(sol_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.startswith('//'):
                    continue
                
                match = re.match(r"pragma solidity (\^?\d+\.\d+\.\d+);", cleaned_line)
                if match:
                    version = match.group(1).replace('^', '')
                    # Handle version ranges by taking the minimum required version
                    return '.'.join(version.split('.')[:3])
        return "0.4.26"  # Default version if no pragma found
    except Exception as e:
        print(f"Version extraction error: {str(e)}")
        return "0.4.26"

def is_solc_installed(version):
    """Check if a specific solc version is installed"""
    try:
        result = subprocess.run(['solc-select', 'versions'], 
                              capture_output=True, text=True)
        return version in result.stdout
    except Exception as e:
        print(f"Version check error: {str(e)}")
        return False

def install_solc_version(version):
    """Install a specific solc version if not present"""
    try:
        print(f"Installing solc {version}...")
        subprocess.run(['solc-select', 'install', version], 
                      check=True, capture_output=True)
        subprocess.run(['solc-select', 'use', version], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed for {version}: {e.stderr}")
        return False

def generate_assembly_for_sol_files(input_folder, output_folder):
    """
    Generate assembly instructions for each Solidity file in the input folder
    and save them in separate folders within the output folder.

    Args:
        input_folder (str): Path to the folder containing Solidity files.
        output_folder (str): Path to the folder where assembly outputs will be saved.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .sol files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".sol"):
            file_path = os.path.join(input_folder, file_name)
            contract_name = os.path.splitext(file_name)[0]
            
            # Extract and verify solc version
            solc_version = extract_solc_version(file_path)
            if not is_solc_installed(solc_version):
                if not install_solc_version(solc_version):
                    print(f"Skipping {file_name} - cannot install required solc {solc_version}")
                    continue
            
            try:
                # Ensure correct version is active
                subprocess.run(['solc-select', 'use', solc_version], 
                              check=True, capture_output=True)
                
                # Use solc to generate assembly instructions
                result = subprocess.run(
                    ["solc", "--asm", file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode == 0:
                    # Save assembly instructions to the file
                    with open(os.path.join(output_folder, f"{contract_name}_assembly.txt"), "w") as asm_file:
                        # Remove all /* */ comments from the assembly output
                        asm_output = re.sub(r"/\*.*?\*/", "", result.stdout, flags=re.DOTALL)
                        asm_output = re.sub(r"\/\/.*?\n", "", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"tag(_\d+:*)", r"tag\n\1\n", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"auxdata.*?\n", "", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"\(", r"\n(\n", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"\)", r"\n)\n", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r",", r"\n", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r":", r"\n", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r" assembly {", "\nassembly\n{", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"=======.*?=======", "", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"EVM assembly.*?\n", "", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"  ", "", asm_output, flags=re.DOTALL)
                        asm_output = re.sub(r"\n\n+", "\n", asm_output, flags=re.DOTALL)
                        # asm_file.write('\n'.join(asm_output.splitlines()[3:]))
                        asm_file.write(asm_output)
                        # asm_file.write(result.stdout)
                    print(f"Assembly for {file_name} saved in {os.path.join(output_folder, f'{contract_name}_assembly.txt')}")
                else:
                    print(f"Error processing {file_name}: {result.stderr}")
                    # Save error to a log file
                    with open(os.path.join(output_folder, f"{contract_name}error.log"), "w") as error_file:
                        error_file.write(result.stderr)

            except subprocess.CalledProcessError as e:
                print(f"Processing failed for {file_name}: {e.stderr}")
            except Exception as e:
                print(f"Unexpected error with {file_name}: {str(e)}")

# Updated input path using your specified location
input_folder_path = r"D:\acadmics\sem 5\Innovation paper\Paper_GNN_CL\cfg implement\dataset\multivul_multiclass_test"
output_folder_path = os.path.join("Bytecode", "Dataset_Assembly")

# Generate assembly instructions
generate_assembly_for_sol_files(input_folder_path, output_folder_path)
