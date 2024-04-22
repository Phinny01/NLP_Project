import os
import re

def find_images_for_study_ids(base_path, positive_study_ids, label):
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
    study_id_label = {}
    study_id_pattern = re.compile(r'\D+(\d+)')  # Regex to find digits after first non-digit characters
    print("here")
    for root, dirs, files in os.walk(base_path):
        match = study_id_pattern.search(os.path.basename(root))
        if match:
            current_study_id = int(match.group(1))  # Convert the found numbers to an integer
            if current_study_id in positive_study_ids:
                if current_study_id not in positive_image_paths:
                    positive_image_paths[current_study_id] = []
                    study_id_label[current_study_id] = label
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        positive_image_paths[current_study_id].append(os.path.join(root, file))

    return positive_image_paths, study_id_label

def find_images_for_study_ids_test(base_path, positive_study_ids_list):
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
    study_id_label = {}
    study_id_pattern = re.compile(r'\D+(\d+)')  # Regex to find digits after first non-digit characters
    for label, positive_study_ids in enumerate(positive_study_ids_list):
    
        

        for root, dirs, files in os.walk(base_path):
            match = study_id_pattern.search(os.path.basename(root))
            if match:
                current_study_id = int(match.group(1))  # Convert the found numbers to an integer
                if current_study_id in positive_study_ids:
                    print("NIMI")
                    if current_study_id not in positive_image_paths:
                        positive_image_paths[current_study_id] = []
                        study_id_label[current_study_id] = label
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            positive_image_paths[current_study_id].append(os.path.join(root, file))

    return positive_image_paths, study_id_label



# Assuming 'training_congruent.csv' has columns 'study_id', 'Pleural Effusion', 'Pneumonia', etc.






# Printing the study_id of positive cases
# print("Study IDs for positive cases of Pleural Effusion and Pneumonia:")
# print(52823782.0 in positive_cases_pn['study_id'].values)
# print(52823782.0 in positive_cases_pf['study_id'].values)






# Now `positive_image_paths` contains the paths to all images for positive study IDs



