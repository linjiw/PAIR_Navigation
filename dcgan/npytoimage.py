# # # import os
# # # import numpy as np

# # # def find_first_npy_shape(folder_path):
# # #     for root, _, files in os.walk(folder_path):
# # #         for file in files:
# # #             if file.endswith(".npy"):
# # #                 data_path = os.path.join(root, file)
# # #                 data = np.load(data_path)
# # #                 print(f"Shape of the first .npy file ({file}):", data.shape)
# # #                 return
# # #     print("No .npy file found in the specified folder.")

# # # folder_path = '/home/linjiw/Downloads/isaacgym-jackal/assets/urdf/jackal/worlds'  # replace with your folder path
# # # find_first_npy_shape(folder_path)
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def save_npy_as_image(folder_path, save_folder):
# #     if not os.path.exists(save_folder):
# #         os.makedirs(save_folder)

# #     for root, _, files in os.walk(folder_path):
# #         for file in files:
# #             if file.endswith(".npy"):
# #                 data_path = os.path.join(root, file)
# #                 data = np.load(data_path)
                
# #                 # Check if the shape is 30x30
# #                 if data.shape == (30, 30):
# #                     # Plot and save as image
# #                     plt.imshow(data, cmap="gray")
# #                     plt.axis('off')
# #                     image_name = os.path.splitext(file)[0] + ".png"
# #                     save_path = os.path.join(save_folder, image_name)
# #                     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
# #                     plt.close()

# # folder_path = '/home/linjiw/Downloads/isaacgym-jackal/assets/urdf/jackal/worlds'  # replace with the path to your folder
# # save_folder = './image_folder'
# # save_npy_as_image(folder_path, save_folder)
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def save_npy_as_image(folder_path, save_folder):
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".npy"):
#                 data_path = os.path.join(root, file)
#                 data = np.load(data_path)
                
#                 # Check if the shape is 30x30
#                 if data.shape == (30, 30):
#                     # Reverse the colors (0 becomes 1 and 1 becomes 0)
#                     data = 1 - data
                    
#                     # Plot and save as image
#                     plt.imshow(data, cmap="gray")
#                     plt.axis('off')
#                     image_name = os.path.splitext(file)[0] + ".png"
#                     save_path = os.path.join(save_folder, image_name)
#                     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#                     plt.close()
# folder_path = '/home/linjiw/Downloads/isaacgym-jackal/assets/urdf/jackal/worlds'  # replace with the path to your folder

# # folder_path = './path_to_your_folder'  # replace with the path to your folder
# save_folder = './image_folder'
# save_npy_as_image(folder_path, save_folder)

import os
import numpy as np
import matplotlib.pyplot as plt

def save_npy_as_image(folder_path, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                data_path = os.path.join(root, file)
                data = np.load(data_path)
                
                # Check if the shape is 30x30
                if data.shape == (30, 30):
                    # Reverse the colors (0 becomes 1 and 1 becomes 0)
                    data = 1 - data

                    # Create a figure with specified figure size and DPI
                    fig = plt.figure(figsize=(0.3, 0.3), dpi=100)  # 0.3 inches x 100 dpi = 30 pixels
                    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0,0)
                    ax.xaxis.set_major_locator(plt.NullLocator())
                    ax.yaxis.set_major_locator(plt.NullLocator())

                    # Plot and save as image
                    ax.imshow(data, cmap="gray")
                    image_name = os.path.splitext(file)[0] + ".png"
                    save_path = os.path.join(save_folder, image_name)
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
folder_path = '/home/linjiw/Downloads/PAIRED-Jackal/dcgan/data/image_folder/train'  # replace with the path to your folder

# folder_path = './path_to_your_folder'  # replace with the path to your folder
save_folder = './../train_img'
save_npy_as_image(folder_path, save_folder)
