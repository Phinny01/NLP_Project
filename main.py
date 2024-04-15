import pandas as pd
import os
import re

def find_images_for_study_ids(base_path, positive_study_ids):
    """
    Traverse the directory structure starting from `base_path` to find images in folders
    that match any of the `positive_study_ids`.

    Args:
    - base_path (str): The base directory to start the search from.
    - positive_study_ids (set): A set of study IDs that are positive for your conditions.

    Returns:
    - dict: A dictionary where keys are study_ids and values are lists of image paths.
    """
    positive_image_paths = {}
    study_id_pattern = re.compile(r'\D+(\d+)')  # Regex to find digits after first non-digit characters

    for root, dirs, files in os.walk(base_path):
        match = study_id_pattern.search(os.path.basename(root))
        if match:
            current_study_id = int(match.group(1))  # Convert the found numbers to an integer
            if current_study_id in positive_study_ids:
                if current_study_id not in positive_image_paths:
                    positive_image_paths[current_study_id] = []
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        positive_image_paths[current_study_id].append(os.path.join(root, file))

    return positive_image_paths



# Assuming 'training_congruent.csv' has columns 'study_id', 'Pleural Effusion', 'Pneumonia', etc.
data = pd.read_csv('training_congruent.csv')

# Filtering for positive cases
positive_cases_pn = set(data[data['Pneumonia'] == 1]['study_id'].astype(int))
positive_cases_pf = set(data[(data['Pleural Effusion'] == 1)]['study_id'].astype(int))
positive_cases_pn = set(data[(data['Pneumothorax'] == 1)]['study_id'].astype(int))
positive_cases_pn = set(data[(data['Atelectasis'] == 1)]['study_id'].astype(int))
positive_cases_pn = set(data[(data['No Finding'] == 1)]['study_id'].astype(int))
base_path = 'train'

# Using the modified function with extracted study IDs
print(find_images_for_study_ids(base_path, positive_cases_pn))





# Printing the study_id of positive cases
# print("Study IDs for positive cases of Pleural Effusion and Pneumonia:")
# print(52823782.0 in positive_cases_pn['study_id'].values)
# print(52823782.0 in positive_cases_pf['study_id'].values)






# Now `positive_image_paths` contains the paths to all images for positive study IDs



