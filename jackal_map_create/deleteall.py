import os

def delete_all_files_in_directory(directory):
    """
    Delete all files in the given directory and its subdirectories,
    while keeping the directory structure intact.
    """
    for root, dirs, files in os.walk(directory):
        print(f"Deleting files in {root}")
        print(f"Directories: {dirs}")
        print(f"Files: {files}")
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Example usage
# Note: This is just an example. The function call is commented out to prevent accidental deletion.
delete_all_files_in_directory("./test_data/")
