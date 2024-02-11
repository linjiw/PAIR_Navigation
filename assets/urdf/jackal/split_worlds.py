import os
import shutil

def copy_unique_files(src_folder, target_folder, check_folders):
    """
    Copies files from src_folder to target_folder if they do not exist in any of the folders in check_folders.

    :param src_folder: Folder to copy files from.
    :param target_folder: Folder to copy files to.
    :param check_folders: Folders to check for file existence.
    """
    # Make sure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate over all files in the source folder
    for file in os.listdir(src_folder):
        src_file_path = os.path.join(src_folder, file)

        # Check if the file is a file and not a directory
        if os.path.isfile(src_file_path):
            # Check if the file does not exist in the check folders
            if all(not os.path.exists(os.path.join(check_folder, file)) for check_folder in check_folders):
                # Copy file to the target folder
                shutil.copy(src_file_path, target_folder)
                print(f"Copied: {file}")

# Example usage
src_folder = './worlds'
target_folder = './worlds_unique'
check_folders = ['./worlds_train', './worlds_test']

copy_unique_files(src_folder, target_folder, check_folders)
