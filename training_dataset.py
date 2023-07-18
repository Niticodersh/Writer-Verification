#importing libraries --------------------------------------------------------------------------------------------------------
import os
import random
import pandas as pd
import numpy as np

#path -----------------------------------------------------------------------------------------------------------------------
train_folder='/content/drive/MyDrive/dataset/train'
path_to_save_training_dataset="/content/drive/MyDrive/dataset/Training_Dataset.csv"

#creating similar and dissimilar pairs --------------------------------------------------------------------------------------
def split_pairs_by_writer_folders(data_folder):
    writer_folders = os.listdir(data_folder)
    similar_pairs = []
    dissimilar_pairs = []

    for folder in writer_folders:
        folder_path = os.path.join(data_folder, folder)
        images = os.listdir(folder_path)

        # Generate all possible pairs of images within the folder
        image_pairs = [(os.path.join(folder_path, images[i]), os.path.join(folder_path, images[j])) for i in range(len(images)) for j in range(i + 1, len(images))]

        # Shuffle the pairs randomly
        random.shuffle(image_pairs)

        # Select 6 random similar pairs and assign label 1
        similar_pairs.extend([(image1_path, image2_path, 1) for image1_path, image2_path in image_pairs[:6]])

        # Select 6 random dissimilar pairs from different writer folders and assign label 0
        other_writer_folders = writer_folders.copy()
        other_writer_folders.remove(folder)

        for _ in range(6):
            random_writer_folder = random.choice(other_writer_folders)
            other_writer_folders.remove(random_writer_folder)

            other_folder_path = os.path.join(data_folder, random_writer_folder)
            other_images = os.listdir(other_folder_path)
            random_image = random.choice(other_images)

            dissimilar_pairs.append((os.path.join(folder_path, random.choice(images)), os.path.join(other_folder_path, random_image), 0))

    return similar_pairs, dissimilar_pairs


data_folder = train_folder 

similar_pairs, dissimilar_pairs = split_pairs_by_writer_folders(data_folder)

print(f"Number of similar writer pairs: {len(similar_pairs)}")
print(f"Number of dissimilar writer pairs: {len(dissimilar_pairs)}")

pairs=[]
pairs.extend(similar_pairs)
pairs.extend(dissimilar_pairs)
pairs

df = pd.DataFrame(pairs, columns=['image1_path', 'image2_path', 'label'])

# Shuffle the samples in the DataFrame
shuffled_df = df.sample(frac=1).reset_index(drop=True)

data=shuffled_df.copy()

# Save the training dataset ------------------------------------------------------------------------------------------------------

data.to_csv(path_to_save_training_dataset,index=False)
