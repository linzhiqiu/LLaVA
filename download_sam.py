import requests
import os
import rarfile

# Define paths
file_path = 'sam.txt'  # Replace with your actual file path
save_directory = './playground/data/sam/'
extract_directory = './playground/data/sam/images/'

# Make sure the save and extract directories exist
os.makedirs(save_directory, exist_ok=True)
os.makedirs(extract_directory, exist_ok=True)

def extract_rar(file_path, extract_to):
    try:
        with rarfile.RarFile(file_path) as opened_rar:
            opened_rar.extractall(extract_to)
        print(f"Extracted {file_path}")
    except rarfile.Error as e:
        print(f"Error extracting {file_path}: {e}")

# Read the file and download each file
with open(file_path, 'r') as file:
    next(file)  # Skip the header line
    for line in file:
        file_name, url = line.strip().split('\t')
        save_path = os.path.join(save_directory, file_name)

        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name}")

            # If the file is a RAR file, extract it
            if file_name.endswith('.rar'):
                extract_rar(save_path, extract_directory)
        else:
            print(f"Failed to download {file_name}")

print("All files downloaded and extracted.")
