# import os
# import numpy as np
# from shutil import copy2

# def split_npy_files(folder_path, train_ratio=0.7):
#     # Step 1: Read the list of .npy files
#     files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

#     # Step 2: Shuffle the files
#     np.random.shuffle(files)

#     # Step 3: Split the files into train and test sets
#     split_index = int(len(files) * train_ratio)
#     train_files = files[:split_index]
#     test_files = files[split_index:]

#     # Step 4: Create train and test folders
#     train_folder = os.path.join(folder_path, 'train')
#     test_folder = os.path.join(folder_path, 'test')
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(test_folder, exist_ok=True)

#     # Step 5: Move the files
#     for file in train_files:
#         copy2(os.path.join(folder_path, file), train_folder)
#     for file in test_files:
#         copy2(os.path.join(folder_path, file), test_folder)

#     print(f"Split completed. Train files: {len(train_files)}, Test files: {len(test_files)}")

# # Usage example:
# split_npy_files('/home/linjiw/Downloads/jackal-map-creation/test_data/grid_files')
import os
import numpy as np
from shutil import move

def reshuffle_train_test(folder_path, train_ratio=0.7):
    # Paths for train and test folders
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')

    # Step 1: Collect all .npy files from train and test folders
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.npy')]
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.npy')]
    all_files = train_files + test_files

    # Step 2: Shuffle the combined files list
    np.random.shuffle(all_files)

    # Step 3: Calculate the split
    split_index = int(len(all_files) * train_ratio)
    new_train_files = all_files[:split_index]
    new_test_files = all_files[split_index:]

    # Step 4: Redistribute the files
    for file in new_train_files:
        if file in test_files:
            move(os.path.join(test_folder, file), os.path.join(train_folder, file))

    for file in new_test_files:
        if file in train_files:
            move(os.path.join(train_folder, file), os.path.join(test_folder, file))

    print(f"Reshuffle completed. Train files: {len(new_train_files)}, Test files: {len(new_test_files)}")

# Usage example:
reshuffle_train_test('/home/linjiw/Downloads/jackal-map-creation/test_data/grid_files')
