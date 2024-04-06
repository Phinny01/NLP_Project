import pandas as pd
from io import StringIO


# Using StringIO to simulate reading from a file
data = pd.read_csv('training_congruent.csv')

# Filtering for positive cases of Pleural Effusion and Pneumonia
positive_cases = data[(data['Pleural Effusion'] == 1) | (data['Pneumonia'] == 1)]

# Printing the study_id of positive cases
print("Study IDs for positive cases of Pleural Effusion and Pneumonia:")
print(positive_cases['study_id'].to_list())
