import pandas as pd
from io import StringIO


# Using StringIO to simulate reading from a file
data = pd.read_csv('training_congruent.csv')

# Filtering for positive cases of Pleural Effusion and Pneumonia
positive_cases_pf = data[(data['Pleural Effusion'] == 1)]
positive_cases_pn = data[(data['Pneumonia'] == 1)]
no_findings = data[(data['No_Findings'] == 1)]
positive_cases_pt = data[(data['Pneumothorax'] == 1)]
positive_cases_at = data[(data['Atelectasis'] == 1)]

# Printing the study_id of positive cases
print("Study IDs for positive cases of Pleural Effusion and Pneumonia:")
print(52823782.0 in positive_cases_pn['study_id'].values)
print(52823782.0 in positive_cases_pf['study_id'].values)



