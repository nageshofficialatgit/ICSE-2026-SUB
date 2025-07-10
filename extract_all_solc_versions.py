import os
import json
import subprocess
import re

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

def main():
	# Path to the dataset folder using Windows-style paths
	dataset_path = 'dataset\\pure_vul_test'
	
	versions = set()
	
	# Process each .sol file in the dataset folder
	for filename in os.listdir(dataset_path):
		if filename.endswith('.sol'):
			sol_file_path = os.path.join(dataset_path, filename)
			
			print(f"Processing: {filename}")
			
			# Extract solc version
			solc_version = extract_solc_version(sol_file_path)
			versions.add(solc_version)

	print(versions)

if __name__ == "__main__":
	main()
