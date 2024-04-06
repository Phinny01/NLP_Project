import pandas as pd
from io import StringIO


# Using StringIO to simulate reading from a file
data = pd.read_csv('training_congruent.csv')

# Filtering for positive cases of Pleural Effusion and Pneumonia
conditions = ['Pleural Effusion', 'Pneumonia', 'No Finding', 'Pneumothorax', 'Atelectasis']
filtered_cases = {}

for condition in conditions:
    filtered_df = data[data[condition] == 1].copy()
    filtered_df['study_id'] = filtered_df['study_id'].astype(int)
    filtered_cases[condition] = filtered_df

# Printing the study_id of positive cases
print("Study IDs for positive cases of Pleural Effusion and Pneumonia:")
print(52823782.0 in filtered_cases['Pneumonia']['study_id'].values)
print(filtered_cases['Pleural Effusion']['study_id'].to_list())



