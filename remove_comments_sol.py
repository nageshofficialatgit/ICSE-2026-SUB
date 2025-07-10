import os
import re

def remove_comments_from_file(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		content = file.read()
	
	# Remove single-line comments (//)
	new_content = re.sub(r'(?://(?!https?://).*)', '', content)
	
	# Remove multi-line comments (/* */)
	new_content = re.sub(r'/\*.*?\*/', '', new_content, flags=re.DOTALL)
	
	# Remove empty lines or lines with just whitespace
	new_content = '\n'.join([line for line in new_content.splitlines() if line.strip()]).replace(' ;', ';')

	# Only overwrite file if content has changed
	if new_content != content:
		with open(file_path, 'w', encoding='charmap') as file:
			file.write(new_content.encode('utf-8').decode('charmap'))

def remove_comments_from_sol_files(folder_path):
	for filename in os.listdir(folder_path):
		if filename.endswith('.sol'):  # Process only .sol files
			file_path = os.path.join(folder_path, filename)
			if os.path.isfile(file_path):
				print(f"Removing comments from: {filename}")
				remove_comments_from_file(file_path)

# Example usage
folder_path = 'dataset\\codes'  # Replace with your folder path
remove_comments_from_sol_files(folder_path)
