import os
import json
import subprocess
import re

# default_version = "0.4.26"
versions = ["0.8.29", "0.7.6", "0.6.12", "0.5.17", "0.4.26"]

version = '0.4.26'
command = ['solc-select', 'use', '0.4.26']
result = subprocess.run(command, capture_output=True, text=True, check=True)

def extract_solc_version(sol_file_path):
	"""Extract the solc version from all pragma statements in the file"""
	try:
		with open(sol_file_path, 'r', encoding='utf-8') as file:
			versions = []
			for line in file:
				cleaned_line = line.strip()
				if not cleaned_line or cleaned_line.startswith('//'):
					continue
				
				# Find all pragma statements with version specifications
				match = re.search(
					r"pragma solidity (\^?|>=?|<=?|>?<?=?)?\s*(\d+\.\d+\.\d+)(?:\s*&&\s*[^;]+)*;", 
					cleaned_line
				)
				if match:
					version = match.group(2)
					versions.append(version)
			
			if not versions:
				raise ValueError("No valid Solidity version pragma found in the file")
			
			# Use the latest version specified in pragma statements
			def parse_version(v):
				return tuple(map(int, v.split('.')))
			
			# Sort versions and pick the highest
			sorted_versions = sorted(versions, key=parse_version, reverse=True)
			return sorted_versions[0]

	except Exception as e:
		print(f"Error extracting Solidity version: {str(e)}")
		return None

def generate_contract_ast(sol_file_path, solc_version):
	"""Generate the AST for the contract using subprocess and solc"""
	global version
	try:
		# Remove hardcoded 0.4.26 logic and use exact version from pragma
		required_version = solc_version

		# Always switch to required version
		if version != required_version:
			command = ['solc-select', 'use', required_version]
			subprocess.run(command, capture_output=True, text=True, check=True)
			version = required_version

		# Run the solc command with the --ast-compact-json flag
		command = ['solc', '--ast-compact-json', sol_file_path]
		result = subprocess.run(command, capture_output=True, text=True, check=True)
		
		# The result of solc is in the stdout; parse it as JSON
		ast_data = json.loads(result.stdout[result.stdout.index('{'):])
		return ast_data
	
	except subprocess.CalledProcessError as e:
		print(f"Error running solc: {e.stderr}")
		return None
	except Exception as e:
		print(f"Error generating AST: {str(e)}")
		return None

def save_ast_data(ast_data, output_path):
	"""Save AST data to a JSON file"""
	try:
		with open(output_path, 'w') as f:
			json.dump(ast_data, f, indent=4)
		
		print(f"AST data saved to: {output_path}")
	except Exception as e:
		print(f"Error saving AST data: {str(e)}")

def main():
	# Path to the dataset folder using Windows-style paths
	# dataset_path = 'dataset\\pure_vul_test'
	dataset_path = 'dataset\\multivul_multiclass_test'
	
	# Create output directories
	output_dir = 'AST\\ast_outputs'
	os.makedirs(output_dir, exist_ok=True)
	
	# Process each .sol file in the dataset folder
	for filename in os.listdir(dataset_path):
		if filename.endswith('.sol'):
			sol_file_path = os.path.join(dataset_path, filename)
			ast_output_path = os.path.join(output_dir, f"{filename[:-4]}_ast.json")
			
			print(f"Processing: {filename}")
			
			# Extract solc version
			solc_version = extract_solc_version(sol_file_path)
			if not solc_version:
				print(f"Failed to extract solc version for {filename}")
				for version in versions:
					# Generate AST
					ast_data = generate_contract_ast(sol_file_path, version)
					
					if ast_data:
						# Save AST data
						save_ast_data(ast_data, ast_output_path)
						break

			else:
				# Generate AST
				ast_data = generate_contract_ast(sol_file_path, solc_version)
				
				if ast_data:
					# Save AST data
					save_ast_data(ast_data, ast_output_path)
				else:
					print(f"Failed to generate AST for {filename}")

if __name__ == "__main__":
	main()
