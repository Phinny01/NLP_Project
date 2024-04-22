import pandas as pd
import keras 
import main
from model import get_images_and_labels
from model import prepare_dataset



print("started")
data = pd.read_csv('test_subset.csv')
label_dict = {'Pneumonia': 0, 'Pleural Effusion': 1, 'Pneumothorax': 2, "Atelectasis": 3, "No Findings": 4}
# Filtering for positive cases
positive_cases_pn = set(data[data['Pneumonia'] == 1]['study_id'].astype(int))
positive_cases_pe = set(data[(data['Pleural Effusion'] == 1)]['study_id'].astype(int))
positive_cases_pt = set(data[(data['Pneumothorax'] == 1)]['study_id'].astype(int))
positive_cases_at = set(data[(data['Atelectasis'] == 1)]['study_id'].astype(int))
positive_cases_nf = set(data[(data['No Finding'] == 1)]['study_id'].astype(int))
# Chaining the | operator to combine all sets

all_cases = [positive_cases_pn, positive_cases_pe]
base_path = 'test'
model_path = '≈model.keras'

AI_model = keras.models.load_model(model_path)

image_dict, study_id_to_label = main.find_images_for_study_ids_test(base_path, all_cases)# Example empty dict, replace with actual dictionary from the function

image_paths, labels = get_images_and_labels(image_dict, study_id_to_label)

dataset = prepare_dataset(image_paths, labels) 

test_loss, test_accuracy = AI_model.evaluate(dataset)

print("Test Loss:", test_loss*100, "%")
print("Test Accuracy:", test_accuracy*100, "%") 
