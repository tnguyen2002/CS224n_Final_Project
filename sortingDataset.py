import os
import shutil


# Set the path of the directory to sort
path_to_sort = 'dataset'

# Iterate through the directories in the specified path
for directory in os.listdir(path_to_sort):
    if os.path.isdir(os.path.join(path_to_sort, directory)):

        # Iterate through the files in the directory
        for file in os.listdir(os.path.join(path_to_sort, directory)):
            # Check if file is .svs or .pdf type
            if not os.path.isdir(os.path.join(path_to_sort, directory,file)):
                shutil.move(os.path.join(path_to_sort, directory, file), os.path.join(path_to_sort))

for directory in os.listdir(path_to_sort):
    if os.path.isdir(os.path.join(path_to_sort, directory)):
        shutil.rmtree(os.path.join(path_to_sort, directory))



# Create a list of all the .pdf files in the directory
pdf_files = [f for f in os.listdir(path_to_sort) if f.endswith('.PDF')]

# Create a dictionary to store the file pairs
file_pairs = {}
prefixes = set()

# Iterate through the .pdf files and find the corresponding .svs files with the same first part of the filename
for pdf_file in pdf_files:
    # Split the filename by the "." character
    file_parts = pdf_file.split('.')
    # Get the first part of the filename
    file_prefix = file_parts[0]
    # print(file_prefix)
    if(file_prefix not in prefixes):
        prefixes.add(file_prefix)
        file_pairs[file_prefix] = [pdf_file]


for file in os.listdir(path_to_sort):
    if file.endswith('.svs'):
        file_parts= file.split('-')
    # Add the file pair to the dictionary
        file_prefix = file_parts[0] + '-' + file_parts[1] + '-' + file_parts[2]
        file_pairs[file_prefix].append(file)

# Iterate through the file pairs and create subdirectories for each pair
for file_prefix, file_pair in file_pairs.items():
    # Create a new subdirectory with the name of the file prefix
    new_directory = os.path.join(path_to_sort, file_prefix)
    os.makedirs(new_directory, exist_ok=True)
    
    # Move the files into the new subdirectory
    for file in file_pair:
        shutil.move(os.path.join(path_to_sort, file), os.path.join(new_directory, file))
        
    # Print a message to confirm the files were moved
    print(f"Moved {file_pair} to {new_directory}")